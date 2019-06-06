import pickle
import numpy as np
from sandbox.envs.maze.maze_evaluate import sample_unif_feas
from sandbox.envs.maze.point_maze_env import PointMazeEnv
from sandbox.logging.visualization import plot_labeled_samples
import argparse

def generate_feasible_states(pkl, maze_id=0, scaling=2):

    env = PointMazeEnv(maze_id=maze_id, maze_size_scaling=scaling)
    feasible_states = sample_unif_feas(env, 10)
    with open(pkl, 'wb') as pkl_file:
        pickle.dump(list(feasible_states), pkl_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_id', type=int, default=0, help="maze id")


    args = parser.parse_args()
    pkl = 'sandbox/young_clgan/envs/maze/feasible_states/feasible%d.pkl' % args.maze_id
    limit = 3 if args.maze_id == 0 else 9 if args.maze_id==14 or args.maze_id == 15 else 11.5 if args.maze_id == 16 or args.maze_id == 18\
        else 11.25 if args.maze_id == 17 else 5
    center = (2, 2) if args.maze_id == 0 else (-4, 4) if args.maze_id == 13 else (2, 0) if args.maze_id == 14 else (-6, 0) \
        if args.maze_id == 15 else (-8, -8) if args.maze_id == 16 or args.maze_id == 18 else (-10.2, -10.2) if args.maze_id == 17 else (0, 0)

    generate_feasible_states(pkl, maze_id = args.maze_id, scaling=1 if args.maze_id==16 else 0.3 if args.maze_id == 17 else 2)

    feasible_states = np.array(pickle.load(open(pkl, 'rb')))
    print(feasible_states.shape)

    plot_labeled_samples(feasible_states, sample_classes=np.array([0] * feasible_states.shape[0]), text_labels={0:''}, markers=None,
                         fname='sandbox/young_clgan/envs/maze/feasible_states/feasible_maze%d.png'%args.maze_id, limit=limit,
                         center=center, size=10000000, colors=('r', 'g', 'b', 'm', 'k'), maze_id=args.maze_id, s=10)



if __name__ == "__main__":
    main()