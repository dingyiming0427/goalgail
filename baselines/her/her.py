import numpy as np
from sandbox.envs.maze.maze_env_utils import find_cell_idx, sample_nearby_states
from sandbox.envs.maze.maze_evaluate import sample_unif_feas

def perturb(states, scale=0.5):
    states += np.random.normal(loc=0.0, scale=scale, size=states.shape)
    return states



def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, nearby_action_penalty=False, nearby_p=0.5, perturb_scale=.5,
                                perturb_to_feasible=False, cells_apart=2, env=None, discriminator=None, gail_weight=0., sample_g_first=False,
                                zero_action_p = 0., dis_bound = np.inf, two_rs=False, with_termination=True):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    try:
        feasible_states = sample_unif_feas(env.wrapped_env, 50)
    except Exception as e:
        print(e)
    def perturb_feasible(states):
        return [sample_nearby_states(env.wrapped_env, coord, feasible_states, num_samples=1, n=0)[0] for coord in states]

    if replay_strategy != 'none':
        future_p = 1 - (1. / (1 + replay_k))

    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        def is_goal_reached(gs, ags):
            dif = gs - ags
            if env.distance_metric == 'L1':
                goal_distance = np.linalg.norm(dif, ord=1, axis=-1)
            elif env.distance_metric == 'L2':
                goal_distance = np.linalg.norm(dif, ord=2, axis=-1)
            elif callable(env.distance_metric):
                goal_distance = env.distance_metric(gs, ags)
            else:
                raise NotImplementedError('Unsupported distance metric type.')

            return goal_distance < env.terminal_eps


        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        successes = episode_batch['successes']
        terminate_idxes = np.argmax(np.array(successes), axis=1)

        terminate_idxes[np.logical_not(np.any(np.array(successes), axis=1))] = T - 1

        episode_idxs = np.random.choice(np.arange(rollout_batch_size), size=batch_size,
                                        p=(terminate_idxes + 1) / np.sum(terminate_idxes + 1))
        # terminate_idxes[terminate_idxes==0] = T-1
        # episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        terminate_idxes = terminate_idxes[episode_idxs]
        # successes = successes[episode_idxs]

        if not sample_g_first:
            t_samples = np.rint(np.random.random(terminate_idxes.shape) * terminate_idxes).astype(int)
        else:
            # import pdb; pdb.set_trace()
            future_t = np.rint(np.random.random(terminate_idxes.shape) * terminate_idxes).astype(int)
            t_samples = np.rint(np.random.random(future_t.shape) * future_t).astype(int)
            her_indexes = np.arange(batch_size)

        assert ((t_samples <= terminate_idxes).all(), t_samples, terminate_idxes)
        # t_samples = np.random.randint(T, size=batch_size)



        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        if not sample_g_first:
            # Select future time indexes proportional with probability future_p. These
            # will be used for HER replay by substituting in future goals.
            her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            # future_offset = np.random.uniform(size=batch_size) * (T - t_samples)\

            future_offset = np.random.uniform(size=batch_size) * (terminate_idxes - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + future_offset)[her_indexes]

            # Replace goal with achieved goal but only for the previously-selected
            # HER transitions (as defined by her_indexes). For the other transitions,
            # keep the original goal.
        if future_p:
            # future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            # transitions['g'][her_indexes] = future_ag
            # transitions['o'][her_indexes] = [env.update_goal_observation(o, ng) for o, ng in
            # zip(transitions['o'][her_indexes], future_ag)]

            future_ag = episode_batch['info_obs2goal'][episode_idxs[her_indexes], future_t]


            if 'relabel_nearby' in replay_strategy:
                # perturb proportion nearby_p of the future_ag to make it a nearby state
                perturb_idx = np.random.choice(future_ag.shape[0], int(future_ag.shape[0] * nearby_p), replace=False)
                if 'onlyfar' in replay_strategy:
                    cell_index_1 = np.array([find_cell_idx(env.wrapped_env, coord) for coord in transitions['ag'][her_indexes][perturb_idx]])
                    cell_index_2 = np.array([find_cell_idx(env.wrapped_env, coord) for coord in future_ag[perturb_idx]])
                    far = np.absolute(cell_index_1 - cell_index_2) > cells_apart
                    perturb_idx = perturb_idx[far]
                future_ag[perturb_idx] = perturb(future_ag[perturb_idx], scale=perturb_scale)

            if nearby_action_penalty:
                if not perturb_to_feasible:
                    future_ag = perturb(future_ag, scale=perturb_scale)
                else:
                    future_ag = perturb_feasible(future_ag)

            transitions['g'][her_indexes] = future_ag
            # transitions['successes'] = is_goal_reached(transitions['g'], transitions['ag_2'])
            if with_termination:
                transitions['successes'] = np.linalg.norm(transitions['g'] - transitions['ag_2'], axis=-1) < env.terminal_eps
            else:
                transitions['successes'] = np.linalg.norm(transitions['g'] - transitions['ag_2'],
                                                      axis=-1) < 1e-6

            if nearby_action_penalty: #TODO: write this as a func
                cell_index_1 = np.array([find_cell_idx(env.wrapped_env, coord) for coord in transitions['ag']])
                cell_index_2 = np.array([find_cell_idx(env.wrapped_env, coord) for coord in transitions['g']])
                transitions['far_from_goal'] = np.absolute(cell_index_1 - cell_index_2) > cells_apart

            # transitions['o'][her_indexes] = [env.update_goal_observation(o, ng) for o, ng in
            #                                                   zip(transitions['o'][her_indexes], future_ag)]
            # transitions['o_2'][her_indexes] = [env.update_goal_observation(o, ng) for o, ng
            #                                                     in zip(transitions['o_2'][her_indexes], future_ag)]


        if zero_action_p > 0:
            zero_action_indexes = np.where(np.random.uniform(size=batch_size) < zero_action_p)
            transitions['o_2'][zero_action_indexes] = transitions['o'][zero_action_indexes]
            transitions['g'][zero_action_indexes] = transitions['ag'][zero_action_indexes]
            transitions['ag_2'][zero_action_indexes] = transitions['ag'][zero_action_indexes]
            transitions['u'][zero_action_indexes] = 0.


        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g', 'o']}
        reward_params['info'] = info
        if discriminator is not None and gail_weight != 0.:
            if two_rs:
                transitions['r2'] = gail_weight * np.clip(
                    (discriminator.get_reward(None, None, None, whole_batch=transitions)),
                    -dis_bound, dis_bound).reshape(-1)
                transitions['r'] = transitions['successes'] * env.goal_weight

            else:
                transitions['r'] = transitions['successes'] * env.goal_weight + \
                                   gail_weight * np.clip((discriminator.get_reward(None, None, None, whole_batch=transitions)),
                                       -dis_bound, dis_bound).reshape(-1)
        else:
            transitions['r'] = transitions['successes'] * env.goal_weight
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
