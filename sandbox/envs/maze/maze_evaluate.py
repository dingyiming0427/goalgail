import argparse
import os

import uuid
import joblib

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
from pylab import *
import pylab
import matplotlib.colorbar as cbar
import matplotlib.patches as patches

from rllab.sampler.utils import rollout
from rllab.misc import logger
from sandbox.envs.base import FixedStateGenerator
# from sandbox.state.selectors import FixedStateSelector
from sandbox.state.evaluator import evaluate_states
from sandbox.logging.visualization import save_image

from sandbox.envs.maze.maze_env_utils import sample_nearby_states

quick_test = False

filename = str(uuid.uuid4())


def get_policy(file):
    policy = None
    train_env = None
    if ':' in file:
        # fetch file using ssh
        os.system("rsync -avrz %s /tmp/%s.pkl" % (file, filename))
        data = joblib.load("/tmp/%s.pkl" % filename)
        if policy:
            new_policy = data['policy']
            policy.set_param_values(new_policy.get_param_values())
        else:
            policy = data['policy']
            train_env = data['env']
    else:
        data = joblib.load(file)
        policy = data['policy']
        train_env = data['env']
    return policy, train_env


def unwrap_maze(env):
    obj = env
    while not hasattr(obj, 'find_empty_space') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    assert hasattr(obj, 'find_empty_space'), "Your train env has not find_empty_spaces!"
    return obj


def sample_unif_feas(train_env, samples_per_cell):
    """
    :param train_env: wrappers around maze
    :param samples_per_cell: how many samples per cell of the maze
    :return: 
    """
    maze_env = unwrap_maze(train_env)
    empty_spaces = maze_env.find_empty_space()

    size_scaling = maze_env.MAZE_SIZE_SCALING

    states = []
    for empty_space in empty_spaces:
        for i in range(samples_per_cell):
            state = np.array(empty_space) + np.random.uniform(-size_scaling/2, size_scaling/2, 2)
            states.append(state)

    return np.array(states)

def my_square_scatter(axes, x_array, y_array, z_array, min_z=None, max_z=None, size=0.5, **kwargs):
    size = float(size)

    if min_z is None:
        min_z = z_array.min()
    if max_z is None:
        max_z = z_array.max()

    normal = pylab.Normalize(min_z, max_z)
    colors = pylab.cm.jet(normal(z_array))

    for x, y, c in zip(x_array, y_array, colors):
        square = pylab.Rectangle((x - size / 2, y - size / 2), size, size, color=c, **kwargs)
        axes.add_patch(square)

    axes.autoscale()

    cax, _ = cbar.make_axes(axes)
    cb2 = cbar.ColorbarBase(cax, cmap=pylab.cm.jet, norm=normal)

    return True

def maze_context(ax, maze_id=0, limit=None, center=None):
    if maze_id == 0:
        ax.add_patch(patches.Rectangle((-3, -3), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 2, 10, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, 5), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -3), 2, 10, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-1, 1), 4, 2, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 11:
        ax.add_patch(patches.Rectangle((-7, 5), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, 1), 10, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 2, 6, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -3), 6, 2, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 12:
        ax.add_patch(patches.Rectangle((-7, 5), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 13:
        ax.add_patch(patches.Rectangle((-3, 1), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((9, -11), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -11), 14, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -11), 2, 14, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-1, -3), 8, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((1, -7), 8, 2, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 15:
        ax.add_patch(patches.Rectangle((-3, -1), 2, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-5, -3), 4, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -7), 2, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-7, -13), 2, 4, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -9), 2, 4, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-3, -13), 2, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((1, -1), 2, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((1, -7), 2, 4, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((5, -3), 2, 4, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((3, -7), 4, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-1, -13), 8, 6, fill=True, edgecolor="none", facecolor='0.4'))

        ax.add_patch(patches.Rectangle((-9, 1), 18, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-9, -15), 2, 18, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-9, -15), 18, 2, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((7, -15), 2, 18, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 16:
        half_size = 5
        length = 2 * half_size + 1
        side = length * 2 + 1
        ax.add_patch(patches.Rectangle((-side + 0.5+3, -length - 0.5+3), half_size, 1, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-side + half_size + 1 + 0.5+3, -length - 0.5+3), length, 1, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-half_size + 0.5+3, -length - 0.5+3), half_size, 1, fill=True, edgecolor="none", facecolor='0.4'))

        ax.add_patch(patches.Rectangle((-length - 0.5+3, -side + 0.5+3), 1, half_size, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-length - 0.5+3, -side + half_size + 1 + 0.5+3), 1, length, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle((-length - 0.5+3, -half_size + 0.5+3), 1, half_size, fill=True, edgecolor="none", facecolor='0.4'))
    elif maze_id == 17:
        scaling = 0.3
        half_size = 18
        length = 2 * half_size + 1
        side = length * 2 + 1
        # half_size *= scaling; length *= scaling; size *= scaling
        ax.add_patch(patches.Rectangle(((-side + 0.5 + 3) * scaling, (-length - 0.5 + 3) * scaling), half_size * scaling, 1 * scaling, fill=True, edgecolor="none",facecolor='0.4'))
        ax.add_patch(patches.Rectangle(((-side + half_size + 1 + 0.5 + 3) * scaling, (-length - 0.5 + 3) * scaling), length * scaling, 1 * scaling, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle(((-half_size + 0.5 + 3) * scaling, (-length - 0.5 + 3) * scaling), half_size * scaling, 1 * scaling, fill=True, edgecolor="none",facecolor='0.4'))

        ax.add_patch(patches.Rectangle(((-length - 0.5 + 3) * scaling, (-side + 0.5 + 3) * scaling), 1 * scaling, half_size * scaling, fill=True, edgecolor="none",facecolor='0.4'))
        ax.add_patch(patches.Rectangle(((-length - 0.5 + 3) * scaling, (-side + half_size + 1 + 0.5 + 3) * scaling), 1 * scaling, length * scaling, fill=True, edgecolor="none", facecolor='0.4'))
        ax.add_patch(patches.Rectangle(((-length - 0.5 + 3) * scaling, (-half_size + 0.5 + 3) * scaling), 1 * scaling, half_size * scaling, fill=True, edgecolor="none", facecolor='0.4'))



    if limit is not None:
        if center is None:
            center = np.zeros(2)
        ax.set_xlim(center[0] - limit, center[0] + limit)
        ax.set_ylim(center[1] - limit, center[1] + limit)


def plot_heatmap(rewards, goals, prefix='', spacing=1, show_heatmap=True, maze_id=0,
                 limit=None, center=None, adaptive_range=False):
    fig, ax = plt.subplots()

    x_goal, y_goal = np.array(goals)[:, :2].T

    if adaptive_range:
        my_square_scatter(axes=ax, x_array=x_goal, y_array=y_goal, z_array=rewards, min_z=np.min(rewards),
                          max_z=np.max(rewards), size=spacing)
    else:
        # THIS IS FOR BINARY REWARD!!!
        my_square_scatter(axes=ax, x_array=x_goal, y_array=y_goal, z_array=rewards, min_z=0, max_z=1, size=spacing)

    maze_context(ax, maze_id=maze_id, limit=limit, center=center)

    ax.scatter(0, 0, color='y', marker='x', s=200)

    # colmap = cm.ScalarMappable(cmap=cm.rainbow)
    # colmap.set_array(rewards)
    # Create the contour plot
    # CS = ax.contourf(xs, ys, zs, cmap=plt.cm.rainbow,
    #                   vmax=zmax, vmin=zmin, interpolation='nearest')
    # CS = ax.imshow([rewards], interpolation='none', cmap=plt.cm.rainbow,
    #                vmax=np.max(rewards), vmin=np.min(rewards)) # extent=[np.min(ys), np.max(ys), np.min(xs), np.max(xs)]
    # fig.colorbar(colmap)

    # ax.set_title(prefix + 'Returns')
    # ax.set_xlabel('goal position (x)')
    # ax.set_ylabel('goal position (y)')

    # ax.set_xlim([np.max(ys), np.min(ys)])
    # ax.set_ylim([np.min(xs), np.max(xs)])
    # plt.scatter(x_goal, y_goal, c=rewards, s=1000, vmin=0, vmax=max_reward)
    # plt.colorbar()
    if show_heatmap:
        plt.show()
    return fig, ax


def test_policy(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1, parallel=True,
                bounds=None, center=None, plot_fail_path=False, plot_rollout=False, horizon = 500, using_gym=False, n_processes=4, feasible_states=None, num_test_samples=20, report=None,
                test_with_nearby=True, plan=None, noise=0):

    if parallel: #and feasible_states is None:
        return test_policy_parallel(policy, train_env, as_goals, visualize, sampling_res, n_traj=n_traj if feasible_states is None else num_test_samples,
                                    center=center, bounds=bounds, return_path=plot_rollout, horizon=horizon,
                                    using_gym=using_gym, n_processes=n_processes, feasible_states=feasible_states, test_with_nearby=test_with_nearby, plan=plan,
                                    noise=noise)

    logger.log("Not using the parallel evaluation of the policy!")
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    states, spacing = find_empty_spaces(train_env, sampling_res=sampling_res)

    old_goal_generator = train_env.goal_generator if hasattr(train_env, 'goal_generator') else None
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None

    if quick_test:
        sampling_res = 0
        empty_spaces = empty_spaces[:3]
        max_path_length = 100
    else:
        max_path_length = horizon

    gen_state_size = np.size(old_goal_generator.state) if old_goal_generator is not None \
        else np.size(old_start_generator)
    avg_totRewards = []
    avg_success = []
    avg_time = []


    for empty_space in states:
        (x, y) = empty_space
        paths = []
        if as_goals:
            goal = (x, y)
            train_env.update_goal_generator(FixedStateGenerator(goal))
        else:
            # init_state = np.zeros_like(old_start_generator.state)
            # init_state[:2] = (x, y)
            init_state = np.pad(empty_space, (0, gen_state_size - np.size(empty_space)), 'constant')
            train_env.update_start_generator(FixedStateGenerator(init_state))
            # print(init_state)
        if feasible_states is None:
            for n in range(n_traj):
                path = rollout(train_env, policy, animated=visualize, max_path_length=max_path_length, speedup=100, using_gym=using_gym, plan=plan)
                paths.append(path)
        else: # used to test local goal reacher
            if test_with_nearby:
                sampled_states = sample_nearby_states(train_env, (x, y), feasible_states, n=3, num_samples=num_test_samples)
            else:
                sampled_states = np.array(feasible_states)[np.random.choice(len(feasible_states), size=num_test_samples)]
            for goal in sampled_states:
                if train_env.relative_goal:
                    train_env.update_goal_generator(FixedStateGenerator(goal - np.array([x, y]))) # relative goals!
                else:
                    train_env.update_goal_generator(FixedStateGenerator(goal))

                # import pdb; pdb.set_trace()
                path = rollout(train_env, policy, max_path_length=max_path_length, using_gym=using_gym)
                if plot_fail_path and np.max(path['env_infos']['goal_reached']) == 0:
                    plot_path(path['env_infos']['xy_pos'], report=report, obs=True, goal=path['env_infos']['goal'][0])
                paths.append(path)
        avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in paths]))
        # avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
        #                                 <= train_env.terminal_eps) for path in paths]))
        avg_success.append(np.mean([np.max(path['env_infos']['goal_reached']) for path in paths]))
        avg_time.append(np.mean([path['rewards'].shape[0] for path in paths]))
    # states = [np.pad(s, (0, gen_state_size - np.size(s)), 'constant') for s in states]

    train_env.update_goal_generator(old_goal_generator)
    train_env.update_start_generator(old_start_generator)

    return avg_totRewards, avg_success, states, spacing, avg_time


def plot_path(paths, num_paths = 1, report = None, obs=False, goal = None, limit=3, center=(0, 0), maze_id=0, plot_action=False, label_ith_state = 0, suffix=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # maze_context(ax, maze_id=maze_id, limit=limit, center=center)
    if num_paths == 1:
        paths = [paths]

    path_lengths = []

    for path in paths:
        if obs:
            observations = path
        else:
            observations = path['observations']
        if goal is None:
            goal = path['env_infos']['goal'][0]


        plt.scatter(observations[0, 0], observations[0, 1], color='b', s=100)
        plt.scatter(observations[-1, 0], observations[-1, 1], color='g', s=100)
        if label_ith_state:
            plt.scatter(observations[label_ith_state, 0], observations[label_ith_state, 1], color='y', s=100)
        if goal is not None:
            plt.scatter(goal[0], goal[1], color='r', marker='*', s = 200)

        plt.scatter(observations[:, 0], observations[:, 1], s=10)

        if plot_action:
            for i, a in enumerate(path['actions']):
                action = limit * a + center
                plt.scatter(action[0], action[1], color='y', s=5 * i)

        path_lengths.append(observations.shape[0])

    ax.set_xlim(center[0] - limit, center[0] + limit)
    ax.set_ylim(center[1] - limit, center[1] + limit)

    vec_img = save_image()
    if report is not None:
        report.add_image(vec_img, 'path length = %d' % path_lengths[0] + '\n' + suffix)


def find_empty_spaces(train_env, sampling_res=1):
    if hasattr(train_env.wrapped_env, 'find_empty_space'):
        maze_env = train_env.wrapped_env
    else:
        maze_env = train_env.wrapped_env.wrapped_env
    empty_spaces = maze_env.find_empty_space()

    size_scaling = maze_env.MAZE_SIZE_SCALING
    num_samples = 2 ** sampling_res
    spacing = size_scaling / num_samples
    starting_offset = spacing / 2

    states = []
    distances = []
    for empty_space in empty_spaces:
        delta_x = empty_space[0]  # - train_env.wrapped_env._init_torso_x
        delta_y = empty_space[1]  # - train_env.wrapped_env._init_torso_y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        distances.append(distance)

    sort_indices = np.argsort(distances)[::-1]

    empty_spaces = np.array(empty_spaces)
    empty_spaces = empty_spaces[sort_indices]
    if quick_test:
        empty_spaces = empty_spaces[:3]

    for empty_space in empty_spaces:
        starting_x = empty_space[0] - size_scaling / 2 + starting_offset
        starting_y = empty_space[1] - size_scaling / 2 + starting_offset
        for i in range(num_samples):
            for j in range(num_samples):
                x = starting_x + i * spacing
                y = starting_y + j * spacing
                states.append((x, y))
    return np.array(states), spacing


def tile_space(bounds, sampling_res=0):
    """sampling_res: how many times split in 2 the axes"""
    assert np.size(bounds[0]) == np.size(bounds[1]), "the bounds are not the same dim!"
    num_samples = 2. ** sampling_res  # num_splits of the axis
    spacing = 1. / num_samples
    starting_offset = spacing / 2

    axes = []
    for idx in range(np.size(bounds[0])):
        axes.append(np.linspace(bounds[0][idx] + starting_offset, bounds[1][idx] - starting_offset,
                                2**sampling_res * (bounds[1][idx] - bounds[0][idx])))
    states = zip(*[g.flat for g in np.meshgrid(*axes)])
    return states, spacing


def test_policy_parallel(policy, train_env, as_goals=True, visualize=True, sampling_res=1, n_traj=1,
                         center=None, bounds=None, return_path=False, horizon=500, using_gym=False, n_processes=4, feasible_states=None, test_with_nearby=True, plan=None,
                         noise=0):
    old_goal_generator = train_env.goal_generator if hasattr(train_env, 'goal_generator') else None
    old_start_generator = train_env.start_generator if hasattr(train_env, 'start_generator') else None
    gen_state_size = np.size(old_goal_generator.state) if old_goal_generator is not None \
                else np.size(old_start_generator)

    if quick_test:
        sampling_res = 0
        max_path_length = 100
    else:
        max_path_length = horizon #originally 400

    if bounds is not None:
        if np.array(bounds).size == 1:
            bounds = [-1 * bounds * np.ones(gen_state_size), bounds * np.ones(gen_state_size)]
        states, spacing = tile_space(bounds, sampling_res)
    else:
        states, spacing = find_empty_spaces(train_env, sampling_res=sampling_res)


    if train_env.relative_goal:
        states = states - train_env.init_goal_obs

    # hack to adjust dim of starts in case of doing velocity also
    states = [np.pad(s, (0, gen_state_size - np.size(s)), 'constant') for s in states]

    avg_totRewards = []
    avg_success = []
    avg_time = []
    logger.log("Evaluating {} states in a grid".format(np.shape(states)[0]))
    import time; before = time.time()
    rewards, paths = evaluate_states(states, train_env, policy, max_path_length, as_goals=as_goals,
                                     n_traj=n_traj, full_path=True, n_processes=n_processes, using_gym=using_gym,
                                     feasible_states=feasible_states, test_with_nearby=test_with_nearby, plan=plan, noise=noise)
    logger.log("States evaluated, time spent: %d" % (time.time() - before))

    path_index = 0
    for _ in states:
        state_paths = paths[path_index:path_index + n_traj]
        avg_totRewards.append(np.mean([np.sum(path['rewards']) for path in state_paths]))
        # avg_success.append(np.mean([int(np.min(path['env_infos']['distance'])
        #                                 <= train_env.terminal_eps) for path in state_paths]))

        avg_success.append(np.mean([np.max(path['env_infos']['goal_reached']) for path in state_paths]))
        avg_time.append(np.mean([path['rewards'].shape[0] for path in state_paths]))

        path_index += n_traj
    if return_path:
        return avg_totRewards, avg_success, states, spacing, avg_time, paths
    return avg_totRewards, avg_success, states, spacing, avg_time


def test_and_plot_policy(policy, env, as_goals=True, visualize=True, sampling_res=1,
                         n_traj=1, max_reward=1, itr=0, report=None, center=None, limit=None, bounds=None, plot_fail_path=False, horizon = 500,
                         using_gym=False, n_processes=4, feasible_states=None, num_test_samples=20, parallel=True, test_with_nearby=True, plot_rollout=False, plan=None,
                         noise=0, log=True):
    obj = env
    while not hasattr(obj, '_maze_id') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    maze_id = obj._maze_id if hasattr(obj, '_maze_id') else None

    ret = test_policy(policy, env, as_goals, visualize, center=center,
                    sampling_res=sampling_res, n_traj=n_traj, bounds=bounds, horizon = horizon,
                    using_gym=using_gym, n_processes=n_processes, feasible_states=feasible_states,
                    num_test_samples=num_test_samples, parallel=parallel, plot_fail_path=(plot_fail_path and itr>=8),
                    report=report, test_with_nearby=test_with_nearby, plan=plan, noise=noise)
    if not plot_rollout:
        avg_totRewards, avg_success, states, spacing, avg_time = ret
    else:
        avg_totRewards, avg_success, states, spacing, avg_time, paths = ret
        # import pdb; pdb.set_trace()
        for path in paths:
            plot_path(path, report=report, limit=limit, center=center, maze_id=maze_id, plot_action=True)


    plot_heatmap(avg_success, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
                 center=center, limit=limit)
    reward_img = save_image()

    # plot_heatmap(avg_time, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
    #              center=center, limit=limit, adaptive_range=True)
    # time_img = save_image()

    mean_rewards = np.mean(avg_totRewards)
    success = np.mean(avg_success)

    if log:
        if not test_with_nearby:
            with logger.tabular_prefix('Outer_global'):
                logger.record_tabular('iter', itr)
                logger.record_tabular('MeanRewards', mean_rewards)
                logger.record_tabular('Success', success)
        else:
            with logger.tabular_prefix('Outer_'):
                logger.record_tabular('iter', itr)
                logger.record_tabular('MeanRewards', mean_rewards)
                logger.record_tabular('Success', success)
        # logger.dump_tabular(with_prefix=False)

    if report is not None:
        report.add_image(
            reward_img,
            'policy performance\n itr: {} \nmean_rewards: {} \nsuccess: {}'.format(
                itr, mean_rewards, success
            )
        )
        # report.add_image(
        #     time_img,
        #     'policy time\n itr: {} \n'.format(
        #         itr
        #     )
        # )
    return mean_rewards, success

def normalize_according_to_max(arr):
    return arr / np.max(np.absolute(arr))

def plot_q_func(policy, env, sampling_res=1, report=None, center=None, limit=None, weight = 1, normalize = False, exponentiate=False, final_goal=None,
                zero_goals=False, discriminator=None, overlay_policy=False, negative=False):  # only for start envs!
    if discriminator is None:
        def get_q(observations, action, goals):
            actions = np.tile(action, (len(observations), 1))
            return policy.sess.run(policy.main.Q_tf, feed_dict={policy.main.o_tf: observations, policy.main.g_tf: goals, policy.main.u_tf: actions})
        name = 'Q'
    else:
        def get_q(observations, action, goals):
            actions = np.tile(action, (len(observations), 1))
            return discriminator.get_reward(observations.astype(np.float32), actions.astype(np.float32), goals.astype(np.float32))
        name = 'D'
    ## For ddpg implemented in openai gym
    HEADWIDTH = 2; HEADLENGTH=3; LINEWIDTHS=0.1; SHAFTWIDTH=0.05
    states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
    if final_goal is not None:
        goal = np.array(final_goal)
    else:
        goal = env.current_goal
    directions = np.array([np.array([0, 1]), np.sqrt([0.5, 0.5]), np.array([1, 0]), np.array([np.sqrt(0.5), -np.sqrt(0.5)]), np.array([0, -1]),
                  -np.sqrt([0.5, 0.5]), np.array([-1, 0]), np.array([-np.sqrt(0.5), np.sqrt(0.5)])])
    if env.relative_goal or (hasattr(env, 'include_maze_obs') and env.include_maze_obs):
        observations = []
        for state in states:
            env.update_start_generator(FixedStateGenerator(state))
            if env.relative_goal:
                env.update_goal_generator(FixedStateGenerator(goal - state))
            else:
                env.update_goal_generator(FixedStateGenerator(goal))

            obs = env.reset()
            observations.append(obs)
        observations = np.array(observations)

    else:
        if env.include_goal_obs:
            if env.append_goal_to_observation:
                observations = np.array([np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal])for state in states])
            else:
                observations = np.array([np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state))]) for state in states])
        else:
            observations = np.array([np.concatenate([[0, ] * (env.observation_space.flat_dim)]) for state in states])

    if overlay_policy:
        actions = policy.get_actions(observations, np.array(states), np.tile(goal, (observations.shape[0], 1)))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(*goal, color='r', marker='*', s=200)
    Qs = np.zeros((observations.shape[0], len(directions)))
    goals = np.tile(goal, (len(observations), 1))
    if hasattr(policy, 'relative_goals') and policy.relative_goals:
        goals = goals -  states
    if zero_goals:
        goals = np.zeros_like(goals)

    for i, dir in enumerate(directions):
        qs = get_q(observations, dir, goals)#  / weight
        Qs[:, i] = qs[:, 0]
        # vecs = dir * qs * 0.2 # scaling with 0.5 to make it more readable
    # if negative:
    #     Qs = - Qs
    normalizing_factor = np.max(np.absolute(Qs)) * 2
    for i, dir in enumerate(directions):
        vecs = dir * np.reshape(Qs[:, i], (-1, 1)) /normalizing_factor
        colors = np.array(['k', 'b'])

        Q = plt.quiver(states[:, 0], states[:, 1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy',
                       scale=1, headwidth = HEADWIDTH, headlength = HEADLENGTH, linewidths = LINEWIDTHS, width = SHAFTWIDTH,
                       color = (colors[(Qs[:, i] < 0).astype(int)]))  # , np.linalg.norm(vars * 4)

        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

    if overlay_policy:
        Q = plt.quiver(states[:, 0], states[:, 1], actions[:, 0], actions[:, 1], units='xy', angles='xy', scale_units='xy', scale=1, color='r')
        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

    if not report:
        plt.show()
    vec_img1 = save_image()
    if report is not None:
        report.add_image(vec_img1, name)


    norms = np.reshape(np.linalg.norm(Qs, axis = 1, ord = 1), (-1, 1))
    exponentiated = np.exp(Qs * 20)
    exp_norm = np.reshape(np.linalg.norm(exponentiated, axis = 1, ord=1), (-1, 1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(*goal, color='r', marker='*', s=200)
    for i, dir in enumerate(directions):
        vecs = dir * np.reshape(Qs[:, i], (-1, 1)) / norms * 4 # normalized
        Q = plt.quiver(states[:, 0], states[:, 1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy', scale=1,
                       headwidth=HEADWIDTH, headlength=HEADLENGTH, linewidths=LINEWIDTHS, width=SHAFTWIDTH)  # , np.linalg.norm(vars * 4)
        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')



    if not report:
        plt.show()
    vec_img2 = save_image()
    if report is not None and normalize:
        report.add_image(vec_img2, name + 'normalized')


    if exponentiate:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(*goal, color='r', marker='*',s=200)
        for i, dir in enumerate(directions):
            vecs = normalize_according_to_max(dir * np.reshape(exponentiated[:, i], (-1, 1)))

            Q = plt.quiver(states[:, 0], states[:, 1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy',
                           scale_units='xy', scale=1, headwidth=HEADWIDTH, headlength=HEADLENGTH, linewidths=LINEWIDTHS, width=SHAFTWIDTH)  # , np.linalg.norm(vars * 4)
            qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

        if overlay_policy:
            Q = plt.quiver(states[:, 0], states[:, 1], actions[:, 0], actions[:, 1], units='xy', angles='xy',
                           scale_units='xy', scale=1, color='r')
            qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

        if not report:
            plt.show()
        vec_img3 = save_image()
        if report is not None:
            report.add_image(vec_img3, name + 'normalized & exponentiated')


def plot_policy_means(policy, env, sampling_res=2, report=None, center=None, limit=None, deterministic = False, GMM = False, final_goal=None, zero_goals=False, suffix=''):  # only for start envs!
    if final_goal is not None:
        goal = final_goal
    else:
        goal = env.current_goal
    if deterministic and not GMM:
        ## For ddpg implemented in openai gym
        states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
        if not (env.relative_goal or (hasattr(env, 'include_maze_obs') and env.include_maze_obs)):
            if env.include_goal_obs:
                if env.append_goal_to_observation:
                    observations = np.array([np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal])for state in states])
                else:
                    observations = np.array([np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state))]) for state in states])
                # observations = [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal]) for state in states]
            else:
                observations = np.array([np.concatenate([[0, ] * (env.observation_space.flat_dim)]) for state in states])
        else: # need reset
            observations = []
            for state in states:
                env.update_start_generator(FixedStateGenerator(state))
                if env.relative_goal:
                    env.update_goal_generator(FixedStateGenerator(goal - state))
                else:
                    env.update_goal_generator(FixedStateGenerator(goal))
                obs = env.reset()
                observations.append(obs)
            observations = np.array(observations)

        if not zero_goals:
            print(observations.shape)
            actions = policy.get_actions(observations, np.array(states), np.tile(goal, (observations.shape[0], 1)))
        else:

            actions = policy.get_actions(observations, np.zeros((observations.shape[0], len(goal))),
                                         np.zeros((observations.shape[0], len(goal))))
        vecs = actions
        # vars = [np.exp(log_std) * 0.25 for log_std in agent_infos['log_std']]
        # ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for e in ells:
        #     ax.add_artist(e)
        #     e.set_alpha(0.2)
        plt.scatter(*goal, color='r', marker='*', s=200)
        Q = plt.quiver(states[:,0], states[:,1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)  # , np.linalg.norm(vars * 4)
        print()
        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
        # cb = plt.colorbar(Q)
        vec_img = save_image()
    elif GMM:
        states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
        # goal = env.current_goal
        observations = [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal])
                        for state in states]  # pad with 0s the obs between CoM position and goal (ie velocity)

        mus, ws = policy.get_actions_means(observations)
        norm_mus = np.linalg.norm(mus, axis = -1)
        normalized_mus = mus/norm_mus[:, :, np.newaxis]

        # vecs = actions
        # vars = [np.exp(log_std) * 0.25 for log_std in agent_infos['log_std']]
        # ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.scatter(*goal, color='r', marker='*',s=200)
        for i in range(normalized_mus.shape[1]):
            Q1 = plt.quiver(states[:, 0], states[:, 1], normalized_mus[:, i, 0] * ws[:, 0] / 2,
                            normalized_mus[:, i, 1] * ws[:, 0] / 2, units='xy', angles='xy', scale_units='xy',
                           scale=1, color = 'r')  # , np.linalg.norm(vars * 4)
            # Q2 = plt.quiver(states[:, 0], states[:, 1], normalized_mus[:, 1, 0] * ws[:, 1] / 2, normalized_mus[:, 1, 1] * ws[:, 1] / 2, units='xy', angles='xy',
            #                scale_units='xy',scale=1, color = 'b')  # , np.linalg.norm(vars * 4)
            # Q3 = plt.quiver(states[:, 0], states[:, 1], normalized_mus[:, 2, 0] * ws[:, 2] / 2, normalized_mus[:, 2, 1] * ws[:, 2] / 2, units='xy', angles='xy',
            #                scale_units='xy',
            #                scale=1, color = 'g')  # , np.linalg.norm(vars * 4)
            # Q4 = plt.quiver(states[:, 0], states[:, 1], normalized_mus[:, 3, 0] * ws[:, 3] / 2, normalized_mus[:, 3, 1] * ws[:, 3] / 2, units='xy', angles='xy',
            #                scale_units='xy',
            #                scale=1)  # , np.linalg.norm(vars * 4)

            print()
            qk = plt.quiverkey(Q1, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q2, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q3, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q4, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
        # cb = plt.colorbar(Q)
        vec_img = save_image()
        report.add_image(vec_img, 'weighted policy mean (normalized)')


        for i in range(normalized_mus.shape[1]):
            Q1 = plt.quiver(states[:, 0], states[:, 1], mus[:, i, 0], mus[:, i, 1], units='xy', angles='xy',
                            scale_units='xy',
                            scale=1, color='r')  # , np.linalg.norm(vars * 4)
            # Q2 = plt.quiver(states[:, 0], states[:, 1], mus[:, 1, 0], mus[:, 1, 1], units='xy', angles='xy',
            #                 scale_units='xy', scale=1, color='b')  # , np.linalg.norm(vars * 4)
            # Q3 = plt.quiver(states[:, 0], states[:, 1], mus[:, 2, 0], mus[:, 2, 1], units='xy', angles='xy',
            #                 scale_units='xy',
            #                 scale=1, color='g')  # , np.linalg.norm(vars * 4)
            # Q4 = plt.quiver(states[:, 0], states[:, 1], mus[:, 3, 0], mus[:, 3, 1], units='xy', angles='xy',
            #                 scale_units='xy',
            #                 scale=1)  # , np.linalg.norm(vars * 4)

            print()
            qk = plt.quiverkey(Q1, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q2, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q3, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # qk = plt.quiverkey(Q4, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
            # cb = plt.colorbar(Q)
        vec_img = save_image()


    else:
        states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
        # goal = env.current_goal
        observations = [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state) - len(goal)), goal])
                        for state in states]
        actions, agent_infos = policy.get_actions(observations)
        vecs = agent_infos['mean']
        vars = [np.exp(log_std) * 0.25 for log_std in agent_infos['log_std']]
        ells = [patches.Ellipse(state, width=vars[i][0], height=vars[i][1], angle=0) for i, state in enumerate(states)]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for e in ells:
            ax.add_artist(e)
            e.set_alpha(0.2)
        plt.scatter(*goal, color='r', marker='*', s=200)
        Q = plt.quiver(states[:,0], states[:,1], vecs[:, 0], vecs[:, 1], units='xy', angles='xy', scale_units='xy', scale=1)  # , np.linalg.norm(vars * 4)
        qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')
        # cb = plt.colorbar(Q)
        vec_img = save_image()



    if report is not None:
        report.add_image(vec_img, 'policy mean' + suffix)

    return states, vecs

def plot_discriminator_heatmap(env, discriminator, expert_policy, goals, report, maze_id, center, limit, agent_policy=None, sampling_res=0, random_actions_per_cell=3, plotname='discriminator heatmap'):

    for goal in goals:
        states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
        observations = np.array(
            [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state))]) for state in states])
        expert_actions = expert_policy.get_actions(observations, np.array(states), np.tile(goal, (observations.shape[0], 1)))

        expert_scores = discriminator.get_reward(observations, expert_actions, np.tile(goal, (observations.shape[0], 1)))
        expert_larger_than_random = np.zeros(shape=(expert_scores.size, random_actions_per_cell))

        for i in range(random_actions_per_cell):
            if agent_policy is not None:
                random_actions = agent_policy.get_actions(observations, np.array(states), np.tile(goal, (observations.shape[0], 1)), noise_eps=0.15)
            else:
                random_actions = np.random.uniform(size=expert_actions.shape, low=-1, high=1)
            random_scores = discriminator.get_reward(observations, random_actions, np.tile(goal, (observations.shape[0], 1)))
            expert_larger_than_random[:, i] = (expert_scores >= random_scores)[:, 0]

        expert_larger_than_random_prob = np.mean(expert_larger_than_random, axis=-1)

        _, ax = plot_heatmap(expert_larger_than_random_prob, states, spacing=spacing, show_heatmap=False, maze_id=maze_id,
                     center=center, limit=limit)

        ax.add_patch(patches.Circle((goal[0], goal[1]), radius=1, fill=True, edgecolor="none", facecolor='#FFFFFF'))

        reward_img = save_image()

        if report is not None:
            report.add_image(
                reward_img,
                '%s \n goal=%s \n avg_success= %f' % (plotname, str(goal), np.mean(expert_larger_than_random_prob))
            )

def plot_discriminator_values(env, discriminator, goal, states, agent_actions, expert_actions, report, maze_id, center, limit):
    # import pdb; pdb.set_trace()
    observations = np.array(
        [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state))]) for state in states])
    expert_scores = discriminator.get_reward(observations, expert_actions, np.tile(goal, (observations.shape[0], 1)))
    agent_scores = discriminator.get_reward(observations, agent_actions, np.tile(goal, (observations.shape[0], 1)))

    vecs_expert = expert_actions / np.linalg.norm(expert_actions, axis=-1, keepdims=True) * expert_scores
    vecs_agent = agent_actions / np.linalg.norm(agent_actions, axis=-1, keepdims=True) * agent_scores

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(*goal, color='r', marker='*', s=200)
    Q = plt.quiver(states[:, 0], states[:, 1], vecs_expert[:, 0], vecs_expert[:, 1], units='xy', angles='xy', scale_units='xy',
                   scale=1, color='r')
    qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

    Q = plt.quiver(states[:, 0], states[:, 1], vecs_agent[:, 0], vecs_agent[:, 1], units='xy', angles='xy',
                   scale_units='xy',
                   scale=1, color='black')
    qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

    vec_img = save_image()


    if report is not None:
        report.add_image(vec_img, 'discriminator scores')

def plot_discriminator_zero_action(env, discriminator, goal, states, report, maze_id, center, limit):
    observations = np.array(
        [np.concatenate([state, [0, ] * (env.observation_space.flat_dim - len(state))]) for state in states])
    actions = np.zeros_like(observations)
    scores = discriminator.get_reward(observations, actions, np.tile(goal, (observations.shape[0], 1)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(*goal, color='r', marker='*', s=200)
    Q = plt.quiver(states[:, 0], states[:, 1], np.zeros_like(scores)[:, 0], scores[:, 0], units='xy', angles='xy',
                   scale_units='xy',
                   scale=1, color='r')
    qk = plt.quiverkey(Q, 0.8, 0.85, 1, r'1 Nkg', labelpos='E', coordinates='figure')

    vec_img = save_image()

    if report is not None:
        report.add_image(vec_img, 'discriminator score with zero action')


def plot_policy_values(env, baseline, sampling_res=2, report=None, center=None, limit=None):  # TODO: try other baseline
    states, spacing = find_empty_spaces(env, sampling_res=sampling_res)
    goal = env.current_goal
    observations = [np.concatenate([state, [0, 0], goal]) for state in states]
    return




def main():
    # pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-20_22-43-48_dav2/log/itr_129/itr_9.pkl"
    #    pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-21_15-30-36_dav2/log/itr_69/itr_4.pkl"
    #    pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-21_22-49-03_dav2/log/itr_199/itr_4.pkl"
    # pkl_file = "sandbox/young_clgan/experiments/point_maze/experiment_data/cl_gan_maze/2017-02-22_13-06-53_dav2/log/itr_119/itr_4.pkl"
    # pkl_file = "data/local/goalGAN-maze30/goalGAN-maze30_2017_02_24_01_44_03_0001/itr_27/itr_4.pkl"
    pkl_file = "/home/davheld/repos/goalgen/rllab_goal_rl/data/s3/goalGAN-maze11/goalGAN-maze11_2017_02_23_01_06_12_0005/itr_199/itr_4.pkl"

    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--file', type=str, default=pkl_file,
    # #                     help='path to the snapshot file')
    # parser.add_argument('--max_length', type=int, default=100,
    #                     help='Max length of rollout')
    # parser.add_argument('--speedup', type=int, default=1,
    #                     help='Speedup')
    # parser.add_argument('--num_goals', type=int, default=200, #1 * np.int(np.square(0.3/0.02))
    #                     help='Number of test goals')
    # parser.add_argument('--num_tests', type=int, default=1,
    #                     help='Number of test goals')
    # args = parser.parse_args()
    #
    # paths = []

    policy, train_env = get_policy(pkl_file)

    avg_totRewards, avg_success, goals, spacing = test_policy(policy, train_env, sampling_res=1)

    plot_heatmap(avg_totRewards, goals, spacing=spacing)


if __name__ == "__main__":
    main()
