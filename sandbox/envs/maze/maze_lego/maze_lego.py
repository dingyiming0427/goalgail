from sandbox.envs.maze.maze_env import MazeEnv
from sandbox.envs.maze.maze_lego.point_lego_env import PointLegoEnv


class LegoMazeEnv(MazeEnv):

    MODEL_CLASS = PointLegoEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True