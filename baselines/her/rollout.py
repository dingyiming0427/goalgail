from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args

import time


def reward_fun(ag_2, g, env):  # vectorized
    if env.distance_metric == 'L1':
        goal_distance = np.linalg.norm(ag_2 - g, ord=1, axis=-1)
    elif env.distance_metric == 'L2':
        goal_distance = np.linalg.norm(ag_2 - g, ord=2, axis=-1)
    elif callable(env.distance_metric):
        goal_distance = env.distance_metric(ag_2 - g, axis=-1)
    else:
        raise NotImplementedError('Unsupported distance metric type.')
    if env.only_feasible:
        return np.logical_and(goal_distance < env.terminal_eps,
                              [env.is_feasible(g_ind) for g_ind in g]) * env.goal_weight
    else:
        return (goal_distance < env.terminal_eps) * env.goal_weight

class RolloutWorker:

    @store_args
    def __init__(self, envs, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, weight = 1, to_goal=None, rollout_terminate=True, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            envs (function): environment instances
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a complterminate_env = terminate_envetely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = envs
        self.rollout_batch_size = len(envs)
        assert self.T > 0
        self.rollout_terminate = rollout_terminate
        if to_goal is None:
            print("to goal is none!")
            self.to_goal = envs[0].transform_to_goal_space
        else:
            self.to_goal = (lambda x: x[to_goal[0]: to_goal[1]]) if len(to_goal) == 2 else (lambda x: x[np.array(to_goal)])

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()



    def reset_rollout(self, i, init_state=None, reset=True):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if reset:
            if init_state is not None:
                obs = self.envs[i].reset(init_state=init_state)
            else:
                obs = self.envs[i].reset()
        else:
            self.envs[i].update_goal()
            obs = self.envs[i].get_current_obs()
        self.initial_o[i] = obs
        self.initial_ag[i] = self.to_goal(obs)
        self.g[i] = self.envs[i].current_goal


    def reset_all_rollouts(self, init_state=None, reset=True):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i, init_state=init_state, reset=reset)

    def generate_rollouts(self, init_state=None, reset=True, slice_goal=None):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """

        self.reset_all_rollouts(init_state=init_state, reset=reset)

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g[:, slice_goal[0]:slice_goal[1]] if slice_goal  is not None else self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, done, info = self.envs[i].step(u[i])
                    # success[i] = info['goal_reached']


                    o_new[i] = curr_o_new
                    ag_new[i] = self.to_goal(curr_o_new)
                    # success[i] = reward_fun(ag_new[i], self.envs[i].current_goal, self.envs[i])[0] / self.weight
                    if self.rollout_terminate:
                        success[i] = int(done)
                    else:
                        success[i] = 0
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                        time.sleep(0.05)
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals,
                       successes=np.array(successes))
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size


        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        import cloudpickle
        with open(path, 'wb') as f:
            cloudpickle.dump(dict(policy=self.policy,
                                  env = self.envs[0]), f, protocol=3)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)


    def envs(self):
        return self.envs
