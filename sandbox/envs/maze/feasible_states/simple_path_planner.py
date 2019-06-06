import numpy as np

from sandbox.envs.maze.maze_env_utils import find_cell_idx
from sandbox.logging.visualization import plot_labeled_samples
from sandbox.envs.maze.point_maze_env import PointMazeEnv

import argparse

def get_locations(env):
    if env._maze_id == 17:
        half_length = 18
    else:
        half_length = 5
    room_length = 2 * half_length + 1
    ROOM_ENTRACE = [(half_length + 1, room_length + 1), # up
                    (room_length + 1 + half_length + 1, room_length + 1), # down
                    (room_length + 1, half_length + 1), #left
                    (room_length + 1, room_length + 1 + half_length + 1), #right
                    ]

    for i in range(4):
        entrance = ROOM_ENTRACE[i]
        if i < 2:
            three_pts = [(entrance[0], entrance[1] - 1), entrance, (entrance[0], entrance[1] + 1)]
        else:
            three_pts = [(entrance[0] - 1, entrance[1]), entrance, (entrance[0] + 1, entrance[1])]
        ROOM_ENTRACE[i] = three_pts

    CORRIDOR_ORDER = np.array([[(2, 0), (3, 0)],
                      [(2, 1), (3, 1)]])

    MID = room_length + 1

    # starting from upper left room, clockwise
    MIDPOINTS = [(), *ROOM_ENTRACE[0], (), *ROOM_ENTRACE[3], (), *ROOM_ENTRACE[1][::-1], (), *ROOM_ENTRACE[2][::-1]]

    return ROOM_ENTRACE, CORRIDOR_ORDER, MID, MIDPOINTS

def get_ele(l, i):
    return l[i % len(l)]


def get_block(env, coord, MIDPOINTS, MID):
    struct = env.coord_to_struct(coord)
    if struct in MIDPOINTS:
        return MIDPOINTS.index(struct)
    else:
        room = (np.array(struct) > MID).astype(int)
        if room[0] == 0:
            return 0 if (room[1] == 0) else 4
        else:
            return 12 if (room[1] == 0) else 8

def get_midpoints_room(env, start, dest):
    ret = [start]
    # start_struct = np.array(env.coord_to_struct(start))
    # dest_struct = np.array(env.coord_to_struct(dest))
    next_midpoint = None
    current = start
    # start_room = (start_struct > MID).astype(int)
    # dest_room = (dest_struct > MID).astype(int)
    # if start_room[0] < dest_room[0]:
    #     ret.append(ROOM_ENTRACE[CORRIDOR_ORDER[start_room[0]][start_room[1]][0]][1])
    # elif start_room[0] > dest_room[0]:
    #     ret.append(ROOM_ENTRACE[CORRIDOR_ORDER[start_room[0]][start_room[1]][0]][::-1][1])
    #
    # start_room[0] = dest_room[0]
    #
    # if start_room[1] < dest_room[1]:
    #     ret.append(ROOM_ENTRACE[CORRIDOR_ORDER[start_room[0]][start_room[1]][1]][1])
    # elif start_room[1] > dest_room[1]:
    #     ret.append(ROOM_ENTRACE[CORRIDOR_ORDER[start_room[0]][start_room[1]][1]][::-1][1])
    #
    # ret = [env.struct_to_coord(struct) for struct in ret]
    while next_midpoint != dest:
        next_midpoint = get_next_midpoint_room(env, current, dest)
        ret.append(next_midpoint)
        current = next_midpoint

    return ret

def get_next_midpoint_room(env, start, dest):
    ROOM_ENTRACE, CORRIDOR_ORDER, MID, MIDPOINTS = get_locations(env)
    start_block = get_block(env, start, MIDPOINTS, MID)
    dest_block = get_block(env, dest, MIDPOINTS, MID)

    if abs(start_block - dest_block) <= 1:
        return dest

    # retstruct = get_ele(MIDPOINTS, start_block + 1) if get_ele(MIDPOINTS, start_block + 1) is not () else get_ele(
    #     MIDPOINTS, start_block + 2)
    if start_block < dest_block:
        if dest_block - start_block <= 8:
            retstruct =  get_ele(MIDPOINTS, start_block + 1) if get_ele(MIDPOINTS, start_block + 1) is not () else get_ele(MIDPOINTS, start_block + 2)
        else:
            retstruct = get_ele(MIDPOINTS, start_block - 1) if get_ele(MIDPOINTS, start_block - 1) is not () else get_ele(MIDPOINTS, start_block - 2)
    if start_block > dest_block:
        if start_block - dest_block > 8:
            retstruct = get_ele(MIDPOINTS, start_block + 1) if get_ele(MIDPOINTS, start_block + 1) is not () else get_ele(MIDPOINTS, start_block + 2)
        else:
            retstruct = get_ele(MIDPOINTS, start_block - 1) if get_ele(MIDPOINTS, start_block - 1) is not () else get_ele(MIDPOINTS, start_block - 2)
    # print(env.struct_to_coord(retstruct))
    return env.struct_to_coord(retstruct)

def get_next_state(env, start, dest, inter_dist):
    next_midpoint = get_next_midpoint_room(env, start, dest)
    interpolated_points = interpolate(start, next_midpoint, inter_dist)
    return interpolated_points[1] if len(interpolated_points) > 2 else next_midpoint

def get_midpoints(env, start, dest):
    """
    :return: start, cell centers along the path, and destination
    """
    start_cell = find_cell_idx(env, start)
    dest_cell = find_cell_idx(env, dest)
    if start_cell == dest_cell:
        return [start, dest]
    diff = dest_cell - start_cell
    ret = env.LINEARIZED[start_cell:dest_cell:diff//abs(diff)]

    # turn struct into coordinate
    ret = [env.struct_to_coord(struct) for struct in ret]
    return [start] + ret[1:] + [dest]

def interpolate(pt1, pt2, inter_dist):
    """
    :param inter_dist: distance between two points interpolated
    :return: a list of points interpolated between two points
    """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    l = np.linalg.norm(pt2 - pt1)
    n = l // inter_dist
    weights = np.arange(0, 1, 1/n).reshape(-1, 1)
    return weights * pt2 + (1 - weights) * pt1


def plan(env, start, dest, inter_dist):
    """
    :param env: a maze environment
    :param start: start position of the agent in the maze
    :param dest: destination position of the agent in the maze
    :return: a list of points along the path
    """
    if env._maze_id == 16:
        midpoints = get_midpoints_room(env, start, dest)
    else:
        midpoints = get_midpoints(env, start, dest)
    ret = []
    for pt1, pt2 in zip(midpoints[:-1], midpoints[1:]):
        ret.extend(interpolate(pt1, pt2, inter_dist))
    return ret + [dest]



def set_goal(path, T):
    """
    :param path: list of points along the path
    :param T: time interval the goal setter works with
    :return: the goal to propose
    """
    if len(path) - 1 < T:
        return path[-1]
    else:
        return path[T]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_id', type=int, default=0, help="maze id")

    args = parser.parse_args()


    env = PointMazeEnv(maze_id=args.maze_id, maze_size_scaling=1 if args.maze_id == 16 else 2)
    start = (0, 0)
    dest = (-15, -15)
    path = np.array(plan(env, start, dest, 0.2))
    print(path)

    limit = 3 if args.maze_id == 0 else 11.5 if args.maze_id == 16 else 5
    center = (2, 2) if args.maze_id == 0 else (-8, -8) if args.maze_id == 16 else (0, 0)

    plot_labeled_samples(path, sample_classes=np.array([0] * path.shape[0]), text_labels={0:''}, markers=None,
                         fname='sandbox/young_clgan/envs/maze/feasible_states/path.png', limit=limit,
                         center=center, size=10000000, colors=('r', 'g', 'b', 'm', 'k'), maze_id=args.maze_id, s=10)