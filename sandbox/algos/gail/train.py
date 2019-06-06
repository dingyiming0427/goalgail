from rllab.misc import logger
from baselines.common.mpi_moments import mpi_moments
import numpy as np
import time
import copy
import random
# from sandbox.logging.logger import ExperimentLogger

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def add_noise_to_action(u, noise_eps=0., max_u=1.):
    noise = noise_eps * max_u * np.random.randn(*u.shape)  # gaussian noise
    u += noise
    u = np.clip(u, -max_u, max_u)
    return u

def relabel_zero_action(batch, zero_action_p):
    batch_size = batch['o'].shape[0]
    zero_action_indexes = np.where(np.random.uniform(size=batch_size) < zero_action_p)
    batch['g'][zero_action_indexes] = transitions['o'][zero_action_indexes] # TODO: may need to transition from obs space to goal space?
    batch['u'][zero_action_indexes] = 0.
    return batch

def train(policy, discriminator, rollout_worker, n_epochs, n_cycles,  n_batches, n_batches_discriminator, normal_buffer, expert_buffer,
          num_rollouts=1, empty_buffer=None, reset=True, feasible_states=None, expert_policy=None, agent_policy=None, train_dis_per_rollout=False,
          noise_agent=0., noise_expert=0., zero_action_p=0., sample_transitions_relabel=None, outer_iter=0, annealing_coeff=1., q_annealing=1., **kwargs):

    def train_discriminator(on_policy_buffer):
        for _ in range(n_batches_discriminator):
            if agent_policy is not None  and feasible_states is not None:
                states = np.array(random.sample(list(feasible_states), batch_size))
                goals = np.array(random.sample(list(feasible_states), batch_size))
                actions = agent_policy.get_actions(states, None, goals, noise_eps=noise_agent)
                batch = {'o': states, 'g': goals, 'u': actions}
            else:
                if on_policy_buffer is not None:
                    batch = on_policy_buffer.sample(batch_size)
                else:
                    batch = normal_buffer.sample(batch_size)
                batch['u'] = add_noise_to_action(batch['u'], noise_eps=noise_agent)

            # query the expert policy directly
            if expert_policy is not None and feasible_states is not None:
                states = np.array(random.sample(list(feasible_states), batch_size))
                goals = np.array(random.sample(list(feasible_states), batch_size))
                actions = expert_policy.get_actions(states, None, goals, noise_eps=noise_expert)
                expert_batch = {'o': states, 'g': goals, 'u': actions}
                if zero_action_p > 0.:
                    expert_batch = relabel_zero_action(expert_batch, zero_action_p)
            else:
                expert_batch = expert_buffer.sample(batch_size)
                expert_batch['u'] = add_noise_to_action(expert_batch['u'], noise_eps=noise_expert)
            discriminator.update(batch, expert_batch)
        return batch, expert_batch

    batch_size = policy.batch_size
    on_policy_buffer = None
    if empty_buffer is not None:
        on_policy_buffer = copy.deepcopy(empty_buffer)

    for epoch in range(n_epochs):
        terminates = []
        nonzero_rew = []
        # train
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            for _ in range(num_rollouts):
                episode = rollout_worker.generate_rollouts(reset=reset)
                policy.store_episode(episode)
                if on_policy_buffer is not None:
                    on_policy_buffer.store_episode(episode)
            start_time = time.time()
            if sample_transitions_relabel is not None:
                old_sample_func = policy.buffer.sample_transitions
                policy.buffer.sample_transitions = sample_transitions_relabel
            for _ in range(n_batches):
                policy.train(annealing_factor=annealing_coeff ** (outer_iter), q_annealing=q_annealing**(outer_iter-1))
                # nonzero_rew.append(np.sum(policy_batch[5] > 0) / len(policy_batch[5]))

            policy.update_target_net()
            if sample_transitions_relabel is not None:
                policy.buffer.sample_transitions = old_sample_func

            if train_dis_per_rollout and n_batches_discriminator > 0 and not (epoch == n_epochs - 1 and cycle == n_cycles - 1):
                batch, expert_batch = train_discriminator(on_policy_buffer)
                if on_policy_buffer is not None:
                    del on_policy_buffer
                    on_policy_buffer = copy.deepcopy(empty_buffer)

            # print("training time: %f" %(time.time() - start_time))
            # successes = episode['successes']
            # terminate_idxes = np.argmax(np.array(successes), axis=1)
            # terminate_idxes[np.logical_not(np.any(np.array(successes), axis=1))] = rollout_worker.T - 1
            # terminates.extend(terminate_idxes)
        if not train_dis_per_rollout and n_batches_discriminator > 0 and epoch != n_epochs - 1:
            batch, expert_batch = train_discriminator(on_policy_buffer)
            if on_policy_buffer is not None:
                del on_policy_buffer
                on_policy_buffer = copy.deepcopy(empty_buffer)


        logger.record_tabular('epoch', epoch)
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if n_batches_discriminator > 0:
            # log avg discriminator output
            logger.record_tabular('d_agent_mean', np.mean(discriminator.get_reward(None, None, None, whole_batch=batch)))
            logger.record_tabular('d_agent_std', np.std(discriminator.get_reward(None, None, None, whole_batch=batch)))
            logger.record_tabular('d_expert_mean', np.mean(discriminator.get_reward(None, None, None, whole_batch=expert_batch)))
            logger.record_tabular('d_expert_std', np.std(discriminator.get_reward(None, None, None, whole_batch=expert_batch)))

        # logger.record_tabular('nonzeroRewardProp', np.mean(np.array(nonzero_rew)))

        logger.dump_tabular()

    return np.mean(np.array(nonzero_rew))



