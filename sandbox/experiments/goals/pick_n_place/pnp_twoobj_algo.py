from sandbox.utils import set_env_no_gpu, format_experiment_prefix

set_env_no_gpu()

import argparse
import math
import os
import os.path as osp
import sys
import random
from multiprocessing import cpu_count
import cloudpickle

import numpy as np
import tensorflow as tf

from sandbox.envs.pick_n_place.pick_n_place_gym import PnPEnv
from sandbox.experiments.goals.pick_n_place.generate_expert_traj_twoobj import collect_demos

from sandbox.state.evaluator import *
# from sandbox.logging.html_report import format_dict, HTMLReport
from sandbox.logging.visualization import *
from sandbox.logging.logger import ExperimentLogger
from rllab.sampler.utils import rollout

from sandbox.algos.gail.train import train
from sandbox.algos.gail.gail import GAIL, Discriminator

import numpy as np
import json
from mpi4py import MPI

from baselines import logger as logger_b
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.ddpg import dims_to_shapes
from baselines.her.replay_buffer import ReplayBuffer
from baselines.her.util import mpi_fork

from subprocess import CalledProcessError

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]




def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])



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


    def make_env(stacking=False):
        return PnPEnv(full_space_as_goal=v['full_space_as_goal'], terminal_eps=0.08, two_obj=True, stacking=stacking,
                      first_in_place=v['first_in_place'])

    env = make_env(stacking=True)
    envs = [env]
    # envs = [env] + [make_env(stacking=True) for _ in range(2)]
    # [e.reset() for e in envs]

    # for _ in range(1000):
    #     env.render()
    #     import pdb; pdb.set_trace()
    #     env.step(env.action_space.sample())

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
    params['batch_size'] = v[
        'batch_size']  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    params['n_test_rollouts'] = v[
        'n_test_rollouts']  # changed from 10 to 3 # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    # exploration
    params['random_eps'] = 0.3  # percentage of time a random action is taken
    params['noise_eps'] = v['action_noise']
    params['goal_weight'] = v['goal_weight']
    params['scope'] = 'ddpg3'

    params['sample_expert'] = v['sample_expert']
    params['expert_batch_size'] = v['expert_batch_size']
    params['bc_loss'] = v['bc_loss']
    params['anneal_bc'] = v['anneal_bc']
    params['gail_weight'] = v['gail_weight']
    params['terminate_bootstrapping'] = v['terminate_bootstrapping']
    params['mask_q'] = int(v['mode'] == 'pure_bc')
    params['two_qs'] = v['two_qs']
    params['anneal_discriminator'] = v['anneal_discriminator']
    params['two_rs'] = v['two_qs'] or v['anneal_discriminator']
    params['with_termination'] = v['rollout_terminate']

    if 'clip_dis' in v and v['clip_dis']:
        params['dis_bound'] = v['clip_dis']

    with open(os.path.join(logger_b.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params['T'] = v['horizon']
    params['to_goal'] = v['to_goal']

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
    params_expert = {k: params[k] for k in ['make_env', 'replay_k', 'discriminator', 'gail_weight', 'two_rs', 'with_termination']}
    params_expert['replay_strategy'] = 'future' if v['relabel_expert'] else 'none'

    params_policy_buffer = {k: params[k] for k in ['make_env', 'replay_k', 'discriminator', 'gail_weight', 'two_rs', 'with_termination']}
    params_policy_buffer['replay_strategy'] = 'future'

    params_empty = {k: params[k] for k in ['make_env', 'replay_k', 'discriminator', 'gail_weight', 'replay_strategy']}

    policy = config.configure_ddpg(dims=dims, params=params, clip_return=v['clip_return'], reuse=tf.AUTO_REUSE, env=env,
                                   to_goal=v['to_goal'])

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': True,
        'T': params['T'],
        'weight': v['goal_weight'],
        'rollout_terminate': v['rollout_terminate'],
        'to_goal': v['to_goal']
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'weight': v['goal_weight'],
        'rollout_terminate': v['rollout_terminate'],
        'to_goal': v['to_goal']
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(envs, policy, dims, logger_b, **rollout_params)
    # rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker([env], policy, dims, logger_b, **eval_params)
    # evaluator.seed(rank_seed)

    n_traj = v['n_evaluation_traj']


    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()
    inner_log_dir = osp.join(log_dir, 'inner_iters')
    # report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)
    # report.add_header("{}".format(EXPERIMENT_TYPE))
    # report.add_text(format_dict(v))

    logger.log("Starting the outer iterations")

    logger.log("Generating heat map")

    def evaluate_pnp(env, policy, outer_iter, thresholds=[0.03, 0.05, 0.08, 0.1, 0.12], threshold_one_part=0.05,
                     n_rollouts=100):
        goal_reached = []
        distance_to_goal = []
        distance_to_goal_hand = []
        distance_to_goal_block = []

        for i in range(n_rollouts):
            traj = rollout(env, policy, max_path_length=v['horizon'], using_gym=True)
            goal_reached.append(np.max(traj['env_infos']['goal_reached']))
            distance_to_goal.append(np.min(traj['env_infos']['distance']))
            if v['full_space_as_goal']:
                distance_to_goal_block.append(np.min(traj['env_infos']['block_distance']))
                distance_to_goal_hand.append(np.min(traj['env_infos']['hand_distance']))

        distance_to_goal = np.array(distance_to_goal)
        distance_to_goal_hand = np.array(distance_to_goal_hand)
        distance_to_goal_block = np.array(distance_to_goal_block)

        logger.record_tabular("Outer_iter", outer_iter)
        # whole goal
        for thr in thresholds:
            success = distance_to_goal < thr
            logger.record_tabular("Outer_Success_%3.2f" % thr, np.mean(success))
        logger.record_tabular("MinDisToGoal", np.mean(distance_to_goal))

        if v['full_space_as_goal']:
            # hand part
            success = distance_to_goal_hand < threshold_one_part
            logger.record_tabular("Success_hand%3.2f" % threshold_one_part, np.mean(success))
            logger.record_tabular("HandMinDisToGoal", np.mean(distance_to_goal_hand))

            # block part
            success = distance_to_goal_block < threshold_one_part
            logger.record_tabular("Success_block%3.2f" % threshold_one_part, np.mean(success))
            logger.record_tabular("BlockMinDisToGoal", np.mean(distance_to_goal_block))

        logger.dump_tabular()
        return np.mean(goal_reached), np.mean(distance_to_goal)

    from sandbox.experiments.goals.pick_n_place.pnp_expert import PnPExpertTwoObj

    expert_policy = PnPExpertTwoObj(env, full_space_as_goal=v['full_space_as_goal'])

    expert_params = {
        'exploit': not v['noisy_expert'],
        'use_target_net': False,
        'use_demo_states': False,
        'compute_Q': False,
        'T': params['T'],
        'weight': v['goal_weight'],
        'rollout_terminate': v['rollout_terminate'],
        'to_goal': v['to_goal']
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        expert_params[name] = params[name]

    expert_params['noise_eps'] = v['expert_noise']
    expert_params['random_eps'] = v['expert_eps']

    expert_worker = RolloutWorker([env], expert_policy, dims, logger_b, **expert_params)

    input_shapes = dims_to_shapes(dims)
    expert_sample_transitions = config.configure_her(params_expert)
    buffer_shapes = {key: (v['horizon'] if key != 'o' else v['horizon'] + 1, *input_shapes[key])
                     for key, val in input_shapes.items()}
    buffer_shapes['g'] = (buffer_shapes['g'][0], 6 if not v['full_space_as_goal'] else 9)
    buffer_shapes['ag'] = (v['horizon'] + 1, 6 if not v['full_space_as_goal'] else 9)
    buffer_shapes['successes'] = (v['horizon'],)
    expert_buffer = ReplayBuffer(buffer_shapes, int(1e6), v['horizon'], expert_sample_transitions)
    policy.expert_buffer = expert_buffer

    sample_transitions_relabel = config.configure_her(params_policy_buffer)



    # for _ in range(v['num_demos']):
        # rollout is generated by expert policy
        # episode = expert_worker.generate_rollouts(reset=not v['no_reset'])
    if v['num_demos'] > 0:
        episode = collect_demos(v['num_demos'], env, exp_noise=v['expert_noise'], exp_eps=v['expert_eps'],
                                render=False, max_path_length=v['horizon'])
        # and is stored into the current expert buffer
        expert_buffer.store_episode(episode)


    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            tf.get_default_session().run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    tf.get_default_session().run(init_new_vars_op)


    # max_success_stacking, min_distance_stacking = evaluate_pnp(env_stacking, policy)
    outer_iter = 0


    max_success, min_distance = evaluate_pnp(env, policy, outer_iter)
    logger.dump_tabular()


    for outer_iter in range(1, v['outer_iters']):
        logger.log("Outer itr # %i" % outer_iter)

        with ExperimentLogger(inner_log_dir, outer_iter, snapshot_mode='last', hold_outter_log=True):
            train(policy, discriminator, rollout_worker, v['inner_iters'], v['n_cycles'], v['n_batches'],
                             v['n_batches_dis'], policy.buffer, expert_buffer,
                             empty_buffer=empty_buffer if v['on_policy_dis'] else None, num_rollouts=v['num_rollouts'],
                             feasible_states=feasible_states if v['query_expert'] else None,
                             expert_policy=expert_policy if v['query_expert'] else None,
                             agent_policy=policy if v['query_agent'] else None,
                             train_dis_per_rollout=v['train_dis_per_rollout'],
                             noise_expert=v['noise_dis_agent'], noise_agent=v['noise_dis_expert'],
                             sample_transitions_relabel=sample_transitions_relabel if v['relabel_for_policy'] else None,
                             outer_iter=outer_iter, annealing_coeff=v['annealing_coeff'], q_annealing=v['q_annealing'],
                             reset=not v['no_reset'])

        print("evaluating policy performance")



        logger.log("Generating heat map")

        success, min_distance = evaluate_pnp(env, policy, outer_iter)

        # logger.record_tabular("Success_Stacking", success_stacking)
        # logger.record_tabular("MinDisToGoal_Stacking", min_distance_stacking)

        logger.dump_tabular()

        if success > max_success:
            print ("% f >= %f, saving policy to params_best" % (success, max_success))
            with open(osp.join(log_dir, 'params_best.pkl'), 'wb') as f:
                cloudpickle.dump({'env': env, 'policy': policy}, f)
            max_success = success

        # report.save()
        # report.new_row()



