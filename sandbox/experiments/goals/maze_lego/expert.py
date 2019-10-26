import numpy as np

BLOCK_SIZE = 1.

def clip_action(ac):
    return np.clip(ac, -1, 1)

class PointLegoExpert:
    def __init__(self, env, full_space_as_goal=False):
        self.env = env

        self.full_space_as_goal = full_space_as_goal

        self.phase = 'horizontal'
        self.subphase = 'goto_pushpoint'

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., **kwargs): # only support one observation
        a = np.array([self.get_action(o[0])])
        noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
        a += noise
        a = np.clip(a, -1, 1)

        a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)

        if a.shape[0] == 1:
            return a[0]
        else:
            return a


        # return np.array([self.env.action_space.sample() for _ in range(len(o))])

    def get_action(self, o, noise=0.1):
        point_pos = o[:2]
        block_pos = o[2:4]
        if not self.full_space_as_goal:
            vec_to_goal = self.env.current_goal - block_pos
        else:
            vec_to_goal = self.env.current_goal[2:4] - block_pos
            vec_to_goal_agent = self.env.current_goal[:2] - point_pos

        # import pdb;
        # pdb.set_trace()
        if abs(vec_to_goal[0]) > 0.05:
            # import pdb; pdb.set_trace()
            if abs(block_pos[0] + BLOCK_SIZE / 2 * (-np.sign(vec_to_goal[0])) - point_pos[0]) > 0.2 or abs(block_pos[1] - point_pos[1]) > 0.2: # if we are far from the push point
                if vec_to_goal[0] > 0: # TODO: this rely on the fact that in the beginning the block is to the right of the point mass
                    ac = np.array([vec_to_goal[0] * 5, 0])
                else: # need to go around the block to the right side: down, right, then up
                    # first see which side of the block we are at
                    vec_to_block = block_pos - point_pos
                    if vec_to_block[0] > 0 and vec_to_block[1] - BLOCK_SIZE / 2 < 0: #if we are to the left side of the block, go down
                        ac = np.array([0, -1])
                    elif vec_to_block[0] + BLOCK_SIZE / 2 + 0.05 > 0 and vec_to_block[1] - BLOCK_SIZE / 2 > 0: # if we are at the downside of the block
                        target_x = block_pos[0] + BLOCK_SIZE / 2 + 0.05
                        ac = np.array([target_x - point_pos[0], 0]) * 5

                    else: # otherwise we can go straight up to the push point
                        # import pdb; pdb.set_trace()
                        target_y = block_pos[1]
                        ac = np.array([0, target_y - point_pos[1]]) * 5
            else:
                ac = np.array([vec_to_goal[0] * 5, 0])

        elif abs(vec_to_goal[1]) > 0.05: # can push vertically now
            if abs(block_pos[1] + BLOCK_SIZE / 2 * (-np.sign(vec_to_goal[1])) - point_pos[1]) > 0.1 and abs(block_pos[0] - point_pos[0]) > 0.1:
                ac = np.array([0, block_pos[1] + BLOCK_SIZE / 2 * (-np.sign(vec_to_goal[1])) - point_pos[1]])
            elif abs(block_pos[0] - point_pos[0]) > 0.1:
                ac = np.array([block_pos[0] - point_pos[0], 0])
            else:
                ac = np.array([0, vec_to_goal[1] * 5])

        else:
            if not self.full_space_as_goal:
                ac = np.array([0, 0])
            else:
                if abs(vec_to_goal_agent[0]) > 0.05:
                    ac = np.array([vec_to_goal_agent[0], 0]) * 5
                elif abs(vec_to_goal_agent[1]) > 0.05:
                    ac = np.array([0, vec_to_goal_agent[1]]) * 5
                else:
                    ac = np.array([0, 0])

        return np.clip(ac, -1, 1.)


    def reset(self):
        pass


    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 2))