from sandbox.envs.maze.maze_evaluate import sample_nearby_states
from sandbox.envs.maze.point_maze_env import PointMazeEnv
from sandbox.logging.visualization import plot_labeled_samples

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--maze_id', type=int, default=0, help="maze id")
args = parser.parse_args()


env = PointMazeEnv(maze_id=args.maze_id)
pkl = 'sandbox/young_clgan/envs/maze/feasible_states/feasible%d.pkl' % args.maze_id
limit = 3 if args.maze_id == 0 else 9 if args.maze_id == 14 else 5
center = (2, 2) if args.maze_id == 0 else (-4, 4) if args.maze_id == 13 else (2, 0) if args.maze_id == 14 else (0, 0)
feasible_states = np.array(pickle.load(open(pkl, 'rb')))

sampled_states = np.array(sample_nearby_states(env, (-4, 0), feasible_states, 100))
plot_labeled_samples(sampled_states, sample_classes=np.array([0] * sampled_states.shape[0]), text_labels={0: ''},
                     markers=None,
                     fname='sandbox/young_clgan/envs/maze/feasible_states/test%d.png' % args.maze_id,
                     limit=limit,
                     center=center, size=10000000, colors=('r', 'g', 'b', 'm', 'k'), maze_id=args.maze_id, s=10)