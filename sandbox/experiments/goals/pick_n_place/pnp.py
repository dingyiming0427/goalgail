from sandbox.utils import set_env_no_gpu, format_experiment_prefix

set_env_no_gpu()

import argparse
import math
import os
import os.path as osp
import sys
import random
from multiprocessing import cpu_count

from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator

from sandbox.experiments.goals.pick_n_place.pnp_algo import run_task


EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='set the additional name for experiment prefix'
    )
    args = parser.parse_args()
    # setup ec2
    ec2_instance = args.type if args.type else 'c4.xlarge'

    # configure instance
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"])  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1

    exp_prefix = 'fetchpnp'
    # default_prefix = 'pushing-her'
    # if args.prefix is None:
    #     exp_prefix = format_experiment_prefix(default_prefix)
    # elif args.prefix == '':
    #     exp_prefix = default_prefix
    # else:
    #     exp_prefix = '{}_{}'.format(default_prefix, args.prefix)

    vg = VariantGenerator()

    vg.add('seed', [10, 20, 30])

    # # GeneratorEnv params
    vg.add('terminal_eps', [0.3])
    vg.add('extend_distance_rew', [False])
    vg.add('terminate_env', [True])

    vg.add('distance_metric', ['L2'])
    vg.add('rollout_terminate', lambda mode: [True] if 'gail' in mode else [False])

    vg.add('policy_save_interval', [5])
    vg.add('clip_return', [1])

    #############################################
    vg.add("lr", [1E-4])
    vg.add("discount", [0.99])

    vg.add('action_l2', [0])
    vg.add('max_u', [1])
    vg.add("replay_strategy", lambda mode: ['none'] if 'gail' in mode else ['future'])
    vg.add("layers", [2])
    vg.add("hidden", [256])
    vg.add("n_cycles", [50])
    vg.add("n_batches", [40])
    vg.add("n_batches_dis", lambda mode: [0] if 'bc' in mode or mode=='her' else [20])
    vg.add("batch_size", [256])
    vg.add("n_test_rollouts", [1])
    vg.add("action_noise", [0.15])
    vg.add("to_goal", lambda full_space_as_goal: [(3, 4, 5)] if not full_space_as_goal else [(0, 1, 2, 3, 4, 5)])


    vg.add('min_reward', lambda goal_weight: [goal_weight * 0.1])  # now running it with only the terminal reward of 1!
    vg.add('max_reward', lambda goal_weight: [goal_weight * 0.9])
    vg.add('horizon', [100])
    vg.add('outer_iters', [1000])
    vg.add('inner_iters', [5])

    vg.add('n_evaluation_traj', [3])

    vg.add('num_demos', lambda mode: [20] if mode == 'pure_bc' else [20] if mode == 'her_bc' else [0] if mode == 'her' else [20])
    vg.add('relabel_expert', lambda num_demos: [True] if num_demos > 0 else [False])
    vg.add('expert_batch_size', lambda mode: [256] if mode == 'pure_bc' else [96])
    vg.add('bc_loss', lambda mode: [1.] if mode == 'pure_bc' else [1.] if mode == 'her_bc' else [0])
    vg.add('anneal_bc', lambda mode: [0] if mode == 'her_bc' else [0])
    vg.add('sample_expert', lambda num_demos: [True] if num_demos > 0 else [False])
    vg.add('annealing_coeff', lambda mode: [0.9] if mode == 'her_bc' else [1.])
    vg.add('noisy_expert', [False])
    vg.add('expert_eps', [0.])
    vg.add('expert_noise', [0.])

    # # gail or gail_her
    vg.add('gail_reward', ['negative'])
    vg.add('on_policy_dis', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('query_expert', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('query_agent', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('train_dis_per_rollout', lambda mode: [True] if 'gail' in mode else [True])
    vg.add('goal_weight', lambda mode: [0] if mode == 'pure_bc' or mode == 'gail' else [0.] if mode == 'gail_her' else [1.])
    vg.add('gail_weight', lambda mode: [0.1] if 'gail' in mode else [0.])  # the weight before gail reward
    vg.add('zero_action_p', lambda mode: [0.] if mode == 'gail' else [0.])
    vg.add('relabel_for_policy', lambda mode: [True] if mode=='gail_her' else [False])
    vg.add('two_qs', lambda mode: [False] if mode == 'gail_her' else [False])
    vg.add('q_annealing', lambda mode: [1.] if mode == 'gail_her' else [1.])
    vg.add('anneal_discriminator', lambda mode: [False] if mode == 'gail_her' else [False])
    vg.add('use_s_p', [True])
    vg.add('only_s', [False])


    vg.add('mode', ['gail_her', 'gail', 'her'])  # pure_bc, her_bc, gail, gail_her, her
    vg.add('num_rollouts', [1])

    vg.add('noise_dis_expert', [0.15])
    vg.add('noise_dis_agent', lambda noise_dis_expert: [noise_dis_expert])

    vg.add('clip_dis', [0.])
    vg.add('terminate_bootstrapping', [True])

    vg.add('full_space_as_goal', [False])


    print('Running {} inst. on type {}, with price {}, parallel {}'.format(
        vg.size, config.AWS_INSTANCE_TYPE,
        config.AWS_SPOT_PRICE, n_parallel
    ))

    for vv in vg.variants():

        if mode in ['ec2', 'local_docker']:

            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode=mode,
                docker_image='yimingding/rllab13-shared3',ƒ
                # Number of parallel workers for sampling
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                # plot=True,
                exp_prefix=exp_prefix,
                # exp_name=exp_name,
                # for sync the pkl file also during the training
                sync_s3_pkl=True,
                # sync_s3_png=True,
                sync_s3_html=True,
                # # use this ONLY with ec2 or local_docker!!!
                pre_commands=[
                    'apt-get --assume-yes update && apt-get --assume-yes install libosmesa6-dev',
                    # 'apt-get install libosmesa6-dev',
                    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin',
                    # 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/code/vendor/mujoco/mjpro150/bin',
                    'export PYTHONPATH=/root/code/rllab/sac:$PYTHONPATH',
                    'export PYTHONPATH=/root/code/rllab/baselines:$PYTHONPATH',
                    'export PYTHONPATH=/root/code/rllab:$PYTHONPATH',
                    'pip install gtimer',
                    'pip install --upgrade gym',
                    # 'export MPLBACKEND=Agg',
                    # 'pip install --upgrade pip',
                    # 'pip install --upgrade -I tensorflow',
                    # 'pip install git+https://github.com/tflearn/tflearn.git',
                    # # 'pip install dominate',
                    # 'pip install multiprocessing_on_dill',
                    'pip install --upgrade scikit-image',
                    'pip install scipy==0.17.0', 
                    # 'conda install numpy -n rllab3 -y',
                ],
            )
            if mode == 'local_docker':
                sys.exit()
        else:
            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode='local',
                n_parallel=n_parallel,
                # Only keep the snapshot parameters for the last iteration
                snapshot_mode="last",
                seed=vv['seed'],
                exp_prefix=exp_prefix,
                print_command=False,
            )
            if args.debug:
                sys.exit()
