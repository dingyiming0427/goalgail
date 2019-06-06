from sandbox.envs.maze.feasible_states import simple_path_planner as planner

import numpy as np

class MazeExpert:
    def __init__(self, env, step_size=0.2):
        self.env = env
        self.step_size = step_size

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., **kwargs): # only support one observation
        o = np.array(o).reshape((-1, 2))
        g = np.array(g).reshape((-1, 2))
        a = np.zeros_like(o)

        for i, (o_i, g_i) in enumerate(zip(o, g)):
            next_state = planner.get_next_state(self.env, o_i, g_i, self.step_size)
            a[i] = (np.array(next_state) - o_i) * 5

        noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
        a += noise
        a = np.clip(a, -1, 1)

        a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)

        # a = np.array(a)

        return a[0] if len(a) == 1 else a

    def get_action(self, o, noise=0.1):
        return self.get_actions([o], None, [self.env.current_goal], noise_eps=noise), None

    def reset(self):
        pass

    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 2))

