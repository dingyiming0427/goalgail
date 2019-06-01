import matplotlib
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

matplotlib.use('Agg')
import os
import os.path as osp
import random
import numpy as np
import cloudpickle

from rllab.misc import logger
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.logging.visualization import save_image, plot_labeled_samples, plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.young_clgan.state.evaluator import convert_label, label_states, evaluate_states, label_states_from_paths, evaluate_state
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from sandbox.young_clgan.state.utils import StateCollection


from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv
from sandbox.young_clgan.envs.maze.maze_evaluate import test_and_plot_policy, sample_unif_feas, unwrap_maze, \
    plot_policy_means, plot_q_func, plot_path, plot_discriminator_heatmap, plot_discriminator_values, plot_discriminator_zero_action
from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv

from sandbox.young_clgan.algos.gail.train import train
from sandbox.young_clgan.algos.gail.gail import GAIL, Discriminator

import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger as logger_b
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.ddpg import dims_to_shapes
from baselines.her.replay_buffer import ReplayBuffer

from subprocess import CalledProcessError


import time
EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]

def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=4)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    if v['control_mode'] == 'linear':
        from sandbox.young_clgan.envs.maze.point_maze_env import PointMazeEnv
        inner_env = normalize(PointMazeEnv(maze_id=v['maze_id'], maze_size_scaling=v['maze_scaling'], control_mode=v['control_mode']))
        inner_env_test = normalize(PointMazeEnv(maze_id=v['maze_id'], maze_size_scaling=v['maze_scaling'], control_mode=v['control_mode']))
    elif v['control_mode'] == 'pos':
        from sandbox.young_clgan.envs.maze.point_maze_pos_env import PointMazeEnv
        inner_env = normalize(PointMazeEnv(maze_id=v['maze_id'], maze_size_scaling=v['maze_scaling'], control_mode=v['control_mode']))
        inner_env_test = normalize(PointMazeEnv(maze_id=v['maze_id'], maze_size_scaling=v['maze_scaling'], control_mode=v['control_mode']))

    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=v['goal_center'])

    num_cpu = 1
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
            print("fancy call succeeded")
        except CalledProcessError:
            print("fancy version of mpi call failed, try simple version")
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()

    # Configure logging
    rank = MPI.COMM_WORLD.Get_rank()
    logdir = ''
    if rank == 0:
        if logdir or logger_b.get_dir() is None:
            logger_b.configure(dir=logdir)
    else:
        logger_b.configure()
    logdir = logger_b.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = v['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    feasible_states = sample_unif_feas(inner_env, 10)
    if v['unif_starts']:
        starts = np.random.permutation(np.array(feasible_states))[:300]
    else:
        starts = np.array([[0, 0]])

    uniform_start_generator = UniformListStateGenerator(
        starts.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'], )


    # Prepare params.
    def make_env(inner_env=inner_env, terminal_eps=v['terminal_eps'], terminate_env=v['terminate_env']):
        return GoalStartExplorationEnv(
            env=inner_env, goal_generator=uniform_goal_generator,
            obs2goal_transform=lambda x: x[:v['goal_size']],
            start_generator=uniform_start_generator,
            obs2start_transform=lambda x: x[:v['goal_size']],
            terminal_eps=terminal_eps,
            distance_metric=v['distance_metric'],
            extend_dist_rew=v['extend_dist_rew'],
            only_feasible=v['only_feasible'],
            terminate_env=terminate_env,
            goal_weight=v['goal_weight'],
            inner_weight=0,
            append_goal_to_observation=False
        )


    env = make_env()
    test_env = make_env(inner_env=inner_env_test, terminal_eps=1., terminate_env=True)

    params = config.DEFAULT_PARAMS
    params['action_l2'] = v['action_l2']
    params['max_u'] = v['max_u']
    params['gamma'] = v['discount']
    params['env_name'] = 'FetchReach-v0'
    params['replay_strategy'] = v['replay_strategy']
    params['lr'] = v['lr']
    params['layers'] = v['layers']
    params['hidden'] = v['hidden']
    params['n_cycles'] = v['n_cycles']  # cycles per epoch
    params['n_batches'] = v['n_batches']  # training batches per cycle
    params['batch_size'] = v['batch_size']  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    params['n_test_rollouts'] = v['n_test_rollouts']  # changed from 10 to 3 # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    # exploration
    params['random_eps'] = 0.3  # percentagcone of time a random action is taken
    params['noise_eps'] = v['action_noise']
    params['goal_weight'] = v['goal_weight']

    with open(os.path.join(logger_b.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params['T'] = v['horizon']
    params['to_goal'] = v['to_goal']
    params['nearby_action_penalty'] = v['nearby_action_penalty']
    params['nearby_penalty_weight'] = v['nearby_penalty_weight']
    params['nearby_p'] = v['nearby_p']
    params['perturb_scale'] = v['perturb_scale']
    params['cells_apart'] = v['cells_apart']
    params['perturb_to_feasible'] = v['perturb_to_feasible']

    params['sample_expert'] = v['sample_expert']
    params['expert_batch_size'] = v['expert_batch_size']
    params['bc_loss'] = v['bc_loss']
    params['anneal_bc'] = v['anneal_bc']
    params['gail_weight']  =v['gail_weight']
    params['terminate_bootstrapping'] = v['terminate_bootstrapping']
    params['mask_q'] = int(v['mode'] == 'pure_bc')
    params['two_qs'] = v['two_qs']
    params['anneal_discriminator'] = v['anneal_discriminator']
    params['two_rs'] = v['two_qs'] or v['anneal_discriminator']
    params['with_termination'] = v['rollout_terminate']


    if 'clip_dis' in v and v['clip_dis']:
        params['dis_bound'] = v['clip_dis']

    params = config.prepare_params(params)
    params['make_env'] = make_env
    config.log_params(params, logger=logger_b)

    dims = config.configure_dims(params)
    # prepare GAIL

    if v['use_s_p']:
        discriminator = GAIL(dims['o'] + dims['o'] + dims['g'] if not v['only_s'] else dims['o'] + dims['g'],
                             dims['o'], dims['o'], dims['g'], 0., gail_loss = v['gail_reward'], use_s_p = True, only_s=v['only_s'])
    else:
        discriminator = GAIL(dims['o'] + dims['u'] + dims['g'] if not v['only_s'] else dims['o'] + dims['g'],
                             dims['o'], dims['u'], dims['g'], 0., gail_loss = v['gail_reward'], only_s=v['only_s'])

    params['discriminator'] = discriminator
    # configure replay buffer for expert buffer
    params_expert = {k:params[k] for k in ['make_env', 'replay_k', 'discriminator', 'gail_weight', 'two_rs', 'with_termination']}
    params_expert['replay_strategy'] = 'future' if v['relabel_expert'] else 'none'
    params_expert['sample_g_first'] = v['relabel_expert'] and v['sample_g_first']
    params_expert['zero_action_p'] = v['zero_action_p']

    params_policy_buffer = {k: params[k] for k in ['make_env', 'replay_k', 'discriminator', 'gail_weight', 'two_rs', 'with_termination']}
    params_policy_buffer['replay_strategy'] = 'future'
    params_policy_buffer['sample_g_first'] = False

    policy = config.configure_ddpg(dims=dims, params=params, clip_return=v['clip_return'], reuse=tf.AUTO_REUSE, env=env)


    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': True,
        'T': params['T'],
        'weight': v['goal_weight'],
        'rollout_terminate': v['rollout_terminate']
    }

    expert_rollout_params = {
        'exploit': not v['noisy_expert'],
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'weight': v['goal_weight'],
        'rollout_terminate': v['rollout_terminate']
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        expert_rollout_params[name] = params[name]

    expert_rollout_params['noise_eps'] = v['expert_noise']
    expert_rollout_params['random_eps'] = v['expert_eps']

    rollout_worker = RolloutWorker([env], policy, dims, logger_b, **rollout_params)

    # prepare expert policy, rollout worker
    import joblib
    if v['expert_policy'] == 'planner':
        from sandbox.young_clgan.experiments.goals.maze.expert.maze_expert import MazeExpert
        expert_policy = MazeExpert(inner_env, step_size=0.2)
    else:
        expert_policy = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), v['expert_policy']))['policy']

    expert_rollout_worker = RolloutWorker([env], expert_policy, dims, logger_b, **expert_rollout_params)
    input_shapes = dims_to_shapes(dims)
    expert_sample_transitions = config.configure_her(params_expert)
    buffer_shapes = {key: (v['horizon'] if key != 'o' else v['horizon'] + 1, *input_shapes[key])
                     for key, val in input_shapes.items()}
    buffer_shapes['g'] = (buffer_shapes['g'][0], 2)
    buffer_shapes['ag'] = (v['horizon'] + 1, 2)
    buffer_shapes['successes'] = (v['horizon'],)
    expert_buffer = ReplayBuffer(buffer_shapes, int(1e6), v['horizon'], expert_sample_transitions)
    policy.expert_buffer = expert_buffer

    sample_transitions_relabel = config.configure_her(params_policy_buffer)

    normal_sample_transitions = policy.sample_transitions
    empty_buffer = ReplayBuffer(buffer_shapes, int(1e6), v['horizon'], normal_sample_transitions)

    if not v['query_expert'] or not 'gail' in v['mode']:
        for i in range(v['num_demos']):
            # rollout is generated by expert policy
            episode = expert_rollout_worker.generate_rollouts(reset=not v['no_resets'])
            # and is stored into the expert buffer
            expert_buffer.store_episode(episode)
            if i <= 20:
                path_length = np.argmax(episode['info_goal_reached'][0])
                path_length = v['horizon'] - 1 if path_length == 0 else path_length

                plot_path(episode['o'][0][:path_length], report=report, obs=True, goal=episode['g'][0][0], limit=v['goal_range'], center=v['goal_center'])
    report.new_row()


    # TODO: what is subsampling_rate
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            tf.get_default_session().run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    tf.get_default_session().run(init_new_vars_op)

    outer_iter = 0
    logger.log('Generating the Initial Heatmap...')



    def evaluate_performance(env):
        four_rooms = np.array([[-2, -2], [-13, -13]])
        if v['unif_starts']:
            mean_rewards, successes = [], []
            for pos in four_rooms:
                env.update_start_generator(FixedStateGenerator(np.array(pos)))
                mr, scs = test_and_plot_policy(policy, env, horizon=v['horizon'],  max_reward=v['max_reward'], sampling_res=sampling_res,
                                               n_traj=v['n_traj'],
                                               itr=outer_iter, report=report, limit=v['goal_range'],
                                               center=v['goal_center'], using_gym=True,
                                               noise=v['action_noise'], n_processes=8, log=False)
                mean_rewards.append(mr)
                successes.append(scs)
            with logger.tabular_prefix('Outer_'):
                logger.record_tabular('iter', outer_iter)
                logger.record_tabular('MeanRewards', np.mean(mean_rewards))
                logger.record_tabular('Success', np.mean(successes))
        else:
            env.update_start_generator(FixedStateGenerator(np.array([0, 0])))
            _, scs = test_and_plot_policy(policy, env, horizon=v['horizon'], max_reward=v['max_reward'], sampling_res=sampling_res,
                                          n_traj=v['n_traj'],
                                          itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'],
                                          using_gym=True,
                                          noise=v['action_noise'], n_processes=8)

        report.new_row()

        env.update_start_generator(uniform_start_generator)

        return scs

    logger.dump_tabular(with_prefix=False)

    import cloudpickle
    max_success = 0.

    if not v['query_expert'] and v['num_demos'] > 0:
        if not v['relabel_expert']:
            goals = goals_filtered = expert_buffer.buffers['g'][:v['num_demos'], 0, :]
        else: # collect all states visited by the expert
            goals = None
            for i in range(v['num_demos']):
                terminate_index = np.argmax(expert_buffer.buffers['successes'][i])
                if np.logical_not(np.any(expert_buffer.buffers['successes'][i])):
                    terminate_index = v['horizon']
                cur_goals = expert_buffer.buffers['o'][i, :terminate_index, :2]
                if goals is None:
                    goals = cur_goals
                else:
                    goals = np.concatenate([goals, cur_goals])
            goal_state_collection = StateCollection(distance_threshold=v['coll_eps'])
            goal_state_collection.append(goals)
            goals_filtered = goal_state_collection.states
    else:
        goals_filtered = goals = np.random.permutation(np.array(feasible_states))[:300]
    if v['agent_state_as_goal']:
        goals = goals
    else:
        feasible_states = sample_unif_feas(inner_env, 10)
        goals = np.random.permutation(np.array(feasible_states))[:300]


    logger.dump_tabular(with_prefix=False)


    for outer_iter in range(1, v['outer_iters']):

        logger.log("Outer itr # %i" % outer_iter)

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            if v['unif_goals']:
                env.update_goal_generator(
                    UniformListStateGenerator(
                        goals.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                    )
                )
            else:
                env.update_goal_generator(FixedStateGenerator(v['final_goal']))

            logger.log("Training the algorithm")

            train(policy, discriminator, rollout_worker, v['inner_iters'], v['n_cycles'], v['n_batches'], v['n_batches_dis'], policy.buffer, expert_buffer,
                  empty_buffer=empty_buffer if v['on_policy_dis'] else None, num_rollouts=v['num_rollouts'], reset=not v['no_resets'],
                  feasible_states=feasible_states if v['query_expert'] else None, expert_policy=expert_policy if v['query_expert'] else None,
                  agent_policy=policy if v['query_agent'] else None, train_dis_per_rollout=v['train_dis_per_rollout'],
                  noise_expert=v['noise_dis_agent'], noise_agent=v['noise_dis_expert'], sample_transitions_relabel=sample_transitions_relabel if v['relabel_for_policy'] else None,
                  q_annealing=v['q_annealing'], outer_iter=outer_iter, annealing_coeff=v['annealing_coeff'])

        # logger.record_tabular('NonZeroRewProp', nonzeros)
        logger.log('Generating the Heatmap...')

        success = evaluate_performance(test_env)

        if success > max_success:
            print ("% f >= %f, saving policy to params_best" % (success, max_success))
            with open(osp.join(log_dir, 'params_best.pkl'), 'wb') as f:
                cloudpickle.dump({'env': env, 'policy': policy}, f)
                max_success = success

        report.new_row()

        logger.dump_tabular(with_prefix=False)
