import numpy as np
import gym

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her import make_sample_her_transitions


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 3.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(5E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg3',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 3, #changed from 10 to 3 # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    # kwargs['make_env'] = make_env

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    #kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals', 'to_goal', 'mask_q',
                 'terminate_bootstrapping', 'two_qs', 'anneal_discriminator']:

        if name in kwargs:
            ddpg_params[name] = kwargs[name]
            kwargs['_' + name] = kwargs[name]
            del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, o, **kwargs):  # vectorized

        if env.relative_goal:
            dif = o[:, -2:]
        else:
            # dif = o[:, :2] - o[:, -2:]
            dif = ag_2 - g
        if env.distance_metric == 'L1':
            goal_distance = np.linalg.norm(dif, ord=1, axis=-1)
        elif env.distance_metric == 'L2':
            goal_distance = np.linalg.norm(dif, ord=2, axis=-1)
        elif callable(env.distance_metric):
            goal_distance = env.distance_metric(ag_2, g)
        else:
            raise NotImplementedError('Unsupported distance metric type.')
        if env.only_feasible:
            ret = np.logical_and(goal_distance < env.terminal_eps, [env.is_feasible(g_ind) for g_ind in g]) * env.goal_weight \
                    - env.extend_dist_rew_weight * goal_distance

        else:
            ret = (goal_distance < env.terminal_eps) * env.goal_weight - env.extend_dist_rew_weight * goal_distance

        return ret
        # return -goal_distance

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k', 'discriminator', 'gail_weight', 'sample_g_first', 'zero_action_p',
                 'dis_bound', 'two_rs', 'with_termination']:
        if name in params:
            her_params[name] = params[name]
            params['_' + name] = her_params[name]
            del params[name]
    if 'nearby_action_penalty' in params:
        sample_her_transitions = make_sample_her_transitions(**her_params, env=env,
                                                         nearby_action_penalty=params['nearby_action_penalty'],
                                                         nearby_p = params['nearby_p'],
                                                         perturb_scale=params['perturb_scale'],
                                                         cells_apart=params['cells_apart'],
                                                         perturb_to_feasible=params['perturb_to_feasible'])
    else:
        sample_her_transitions = make_sample_her_transitions(**her_params, env=env)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=1, env=None, to_goal=None, logger=None):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    # env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': False,  # clip positive returns
                        #'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'clip_return': params['goal_weight'] if clip_return == 1 else (1. / (1. - gamma)) * params['goal_weight'] if clip_return == 2 else np.inf,
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'env': env,
                        'to_goal': to_goal,

                        })
    if 'sample_expert' in params:
        ddpg_params.update({
            'sample_expert': params['sample_expert'],
            'expert_batch_size': params['expert_batch_size'],
            'bc_loss': params['bc_loss'],
            'anneal_bc': params['anneal_bc'],
        })
    if 'nearby_action_penalty' in params:
        ddpg_params.update({
            'nearby_action_penalty': params['nearby_action_penalty'],
            'nearby_penalty_weight': params['nearby_penalty_weight'],
        })

    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs.shape[0],
        'u': env.action_space.shape[0],
        'g': len(env.current_goal),
    }

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
