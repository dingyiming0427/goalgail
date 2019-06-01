import os
import random

# os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import tflearn
import argparse
import sys
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from sandbox.carlos_snn.autoclone import autoclone
from rllab import config

from sandbox.young_clgan.experiments.goals.maze.maze_her_gail_algo import run_task

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
    args = parser.parse_args()

    if args.clone:
        autoclone.autoclone(__file__, args)

#   setup ec2
    subnets = ['us-west-1b', 'us-west-1c']
    # subnets = [
    #     'ap-southeast-1a', 'ap-southeast-1b', 'ap-northeast-1a',
    #     'us-east-1d', 'us-east-1a', 'us-east-1b', 'eu-west-1b', 'eu-west-1c', 'eu-west-1a'
    # ]
    # subnets = [
    #     'us-west-2a', 'us-west-2c'
    # ]
    ec2_instance = args.type if args.type else 'c4.xlarge'
    # configure instan
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    # n_parallel = 1
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1
        # n_parallel = -1
        # n_parallel = multiprocessing.cpu_count()

    exp_prefix = 'fourroom'

    vg = VariantGenerator()
    vg.add('maze_id', [16])  # default is 0
    vg.add('maze_scaling', lambda maze_id: [1] if maze_id == 16 or maze_id == 18 else [0.3] if maze_id == 17 else [2])

    vg.add('ultimate_goal', lambda maze_id: [(0, 4)] if maze_id == 0 else [(2, 4), (0, 0)] if maze_id == 12 else [(4, 4)]) #originally 4, 4
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('terminal_eps', [1])
    vg.add('only_feasible', [False])
    vg.add('goal_range',
           lambda maze_id: [3] if maze_id == 0 else [7] if maze_id == 14 or maze_id == 15 else [11.5] \
               if maze_id == 16 or maze_id== 18 else [11.25] if maze_id == 17 else [5])  # this will be used also as bound of the state_space
    # vg.add('goal_range',
    #        lambda maze_id: [4] if maze_id == 0 else [7])  # this will be used also as bound of the state_space
    vg.add('goal_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, -6)] if maze_id == 15 else [(-8, -8)]  \
            if maze_id == 16 or maze_id == 18 else [(-10.2, -10.2)] if maze_id == 17 else [(0, 0)])
    # brownian params
    vg.add('seed_with', ['only_goods'])  # good from brown, onPolicy, previousBrown (ie no good)
    vg.add('brownian_variance', [1])
    vg.add('brownian_horizon', [50])
    # goal-algo params
    vg.add('min_reward', [0.1])
    vg.add('max_reward', [0.9])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [0.])  # put a positive number for a negative reward!
    vg.add('persistence', [1])
    vg.add('n_traj', [1])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    vg.add('sampling_res', lambda maze_id: [0] if maze_id == 0 else [0])
    vg.add('with_replacement', [False])
    vg.add('use_trpo_paths', [False])

    vg.add('unif_goals', [True])  # put False for fixing the goal below!
    vg.add('final_goal', lambda maze_id: [(0, 4)] if maze_id == 0 else [(2, 4), (0, 0)] if maze_id == 12 else [(4, 4)])
    vg.add('unif_starts', [False])
    vg.add('no_resets', lambda unif_starts: [False] if unif_starts else [False])

    # replay buffer
    vg.add('replay_buffer', [True])
    vg.add('coll_eps', [0.5])
    vg.add('num_new_goals', [200])
    vg.add('num_old_goals', [100])
    # sampling paramsherdebug
    vg.add('horizon', lambda control_mode: [500] if control_mode == 'linear' else [300])
    vg.add('terminate_env', [True])
    vg.add('rollout_terminate', [True])
    vg.add('outer_iters', lambda maze_id: [200] if maze_id == 0 else [1000])
    vg.add('inner_iters', [5])  # again we will have to divide/adjust the

    vg.add("lr", [1E-4])
    vg.add("discount", lambda horizon: [1 - 1. / horizon])


    vg.add('action_l2', [0])
    vg.add('max_u', [1])
    vg.add("replay_strategy", lambda mode: ['future'] if 'gail' not in mode else ['none'])
    vg.add("layers", [2])
    vg.add("hidden", [256])
    vg.add("n_cycles", [50])
    vg.add("n_batches", [200])
    vg.add("n_batches_dis", lambda mode: [0] if 'bc' in mode or mode=='her' else [100])
    vg.add("batch_size", lambda mode: [0] if mode == 'pure_bc' else [256])
    vg.add("n_test_rollouts", [1])
    vg.add("action_noise", [0.15])
    vg.add("to_goal", [(0, 2)])

    vg.add('nearby_action_penalty', [False])
    vg.add('nearby_penalty_weight', lambda nearby_action_penalty: [0.001] if nearby_action_penalty else [0])
    vg.add('nearby_p', lambda replay_strategy: [1.] if replay_strategy == 'relabel_nearby' else [0.8] if replay_strategy=='relabel_nearby_onlyfar' else [0.])
    vg.add('perturb_scale', [0.2])
    vg.add('cells_apart', [8])
    vg.add('perturb_to_feasible', [True])

    vg.add('seed', range(500, 1000, 100))

    # expert
    vg.add('num_demos', lambda mode: [20] if mode == 'pure_bc' else [20] if mode == 'her_bc' else [0] if mode == 'her' else [20])
    vg.add('expert_policy', lambda control_mode: ['expert/maze16/params_best.pkl'] if control_mode == 'linear' else ['planner'])#'expert/maze16/params_quasi.pkl'])
    vg.add('relabel_expert', lambda num_demos: [True] if num_demos > 0 else [False])
    vg.add('sample_g_first', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('noisy_expert', [False])
    vg.add('expert_eps', [0.])
    vg.add('expert_noise', [0.])


    # bc or bc_her
    vg.add('expert_batch_size', lambda mode: [256] if mode == 'pure_bc' else [96])
    vg.add('bc_loss', lambda mode: [1.] if mode == 'pure_bc' else [0.1] if mode == 'her_bc' else [0])
    vg.add('anneal_bc', lambda mode: [0] if mode == 'her_bc' else [0])
    vg.add('sample_expert', lambda mode: [True] if 'bc' in mode or 'gail' in mode else [False])
    vg.add('annealing_coeff', lambda mode: [0.9] if mode == 'her_bc' else [1.])

    # gail or gail_her
    vg.add('gail_reward', ['negative'])
    vg.add('on_policy_dis', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('query_expert', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('query_agent', lambda mode: [False] if 'gail' in mode else [False])
    vg.add('train_dis_per_rollout', lambda mode: [True] if 'gail' in mode else [True])
    vg.add('goal_weight', lambda mode: [0] if mode == 'pure_bc' or mode == 'gail' else [0.] if mode == 'gail_her' else [1.])
    vg.add('gail_weight', lambda mode: [0.1] if 'gail' in mode else [0.]) # the weight before gail reward
    vg.add('zero_action_p', lambda mode: [0.] if mode == 'gail' else [0.])
    vg.add('relabel_for_policy', lambda mode: [True] if mode == 'gail_her' else [False])
    vg.add('two_qs', lambda mode: [False] if mode == 'gail_her' else [False])
    vg.add('q_annealing', lambda mode: [1.] if mode == 'gail_her' else [1.])
    vg.add('anneal_discriminator', lambda mode: [False] if mode == 'gail_her' else [False])
    vg.add('use_s_p', [True])
    vg.add('only_s', [False])

    vg.add('mode', ['gail', 'gail_her', 'her']) # pure_bc, her_bc, gail, gail_her, her
    vg.add('num_rollouts', [1])
    vg.add('agent_state_as_goal', lambda mode: [False] if mode != 'gail' else [False])
    vg.add('control_mode', ['pos'])

    vg.add('noise_dis_expert', [0.15])
    vg.add('noise_dis_agent', lambda noise_dis_expert: [noise_dis_expert])

    vg.add('clip_dis', [0.])
    vg.add('terminate_bootstrapping', [True])

    vg.add('policy_save_interval', [5])
    vg.add('clip_return', [1.])

    # Launching
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel), subnets)

    for vv in vg.variants():
        if args.debug:
            run_task(vv)

        if mode in ['ec2', 'local_docker']:
            # choose subnet
            subnet = random.choice(subnets)
            config.AWS_REGION_NAME = subnet[:-1]
            config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[
                config.AWS_REGION_NAME]
            config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[
                config.AWS_REGION_NAME]
            config.AWS_SECURITY_GROUP_IDS = \
                config.ALL_REGION_AWS_SECURITY_GROUP_IDS[
                    config.AWS_REGION_NAME]
            config.AWS_NETWORK_INTERFACES = [
                dict(
                    SubnetId=config.ALL_SUBNET_INFO[subnet]["SubnetID"],
                    Groups=config.AWS_SECURITY_GROUP_IDS,
                    DeviceIndex=0,
                    AssociatePublicIpAddress=True,
                )
            ]

            run_experiment_lite(
                # use_cloudpickle=False,
                stub_method_call=run_task,
                variant=vv,
                mode=mode,
                docker_image='yimingding/rllab13-shared3',
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
                    # 'export MPLBACKEND=Agg',
                    # 'pip install --upgrade pip',
                    # 'conda install mkl',
                    # 'pip install git+https://github.com/tflearn/tflearn.git',
                    # # 'pip install dominate',
                    # 'pip install multiprocessing_on_dill',
                    'pip install scikit-image',
                    # 'conda install numpy -n rllab3 -y',
                ],
            )
            # print(n_parallel)
            # print("got to the end of run_experiment_lite!")
            if mode == 'local_docker':
                sys.exit()
        else:
            print(n_parallel)
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
                # exp_name=exp_name,
            )
