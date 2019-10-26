import numpy as np

def clip_action(ac):
    return np.clip(ac, -1, 1)

class PnPExpert:
    def __init__(self, env, step_size=0.03, full_space_as_goal=False):
        self.env = env
        self.step_size = step_size

        self.full_space_as_goal = full_space_as_goal


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
        gripper_pos = o[:3]
        block_pos = o[3:6]
        current_goal_block = self.env.current_goal[-3:]
        if np.linalg.norm(block_pos - current_goal_block) < 0.04 and self.full_space_as_goal and np.linalg.norm(current_goal_block - self.env.current_goal[:3]) > 0.0001:
            if np.linalg.norm(o[6:9]) < 0.05:
                ac = np.array([0., 0., 1., 1.])
            else:
                ac = self.goto_goal(gripper_pos, self.env.current_goal[:3], grip=1)
        elif np.linalg.norm(o[6:9]) > 0.05 \
            and np.linalg.norm(o[6:9] - np.array([0, 0, 0.03])) > 0.001: #TODO: is it possible it's very close at the beginning
            ac =  self.goto_block(gripper_pos, block_pos)
        elif o[9] < 0.025 and np.linalg.norm(gripper_pos - block_pos) < 0.01:
            ac =  self.goto_goal(block_pos, current_goal_block)

        else:
            ac =  self.pickup_block(gripper_pos, block_pos, o[9])

        return ac



    def reset(self):
        pass


    def goto_block(self, cur_pos, block_pos):
        target_pos = block_pos + np.array([0, 0, 0.03])

        ac = clip_action((target_pos - cur_pos) / self.step_size)

        return np.concatenate([ac, np.array([1])])

    def pickup_block(self, cur_pos, block_pos, gripper_state):
        if np.linalg.norm(cur_pos - block_pos) < 0.01: #and gripper_state > 0.025: # TODO: need to adjust
            ac = np.array([0, 0, 0, -1.])
        else:
            ac = clip_action((block_pos - cur_pos) / self.step_size)
            ac = np.concatenate([ac, np.array([0.])])

        return ac


    def goto_goal(self, cur_pos, goal_pos, grip=-1):
        target_pos = goal_pos

        ac = clip_action((target_pos - cur_pos) / self.step_size)

        return np.concatenate([ac, np.array([grip])])



    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 4))


class PnPExpertTwoObj:
    def __init__(self, env, step_size=0.03, full_space_as_goal=False):
        self.env = env
        self.step_size = step_size

        self.full_space_as_goal = full_space_as_goal


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
        def deal_with_block(block_pos, current_goal_block, gripper_pos, relative_block_to_gripper):
            if np.linalg.norm(relative_block_to_gripper) > 0.1 \
                    and np.linalg.norm(
                relative_block_to_gripper - np.array([0, 0, 0.08])) > 0.001:  # TODO: is it possible it's very close at the beginning
                ac = self.goto_block(gripper_pos, block_pos)
            elif o[15] < 0.025 and np.linalg.norm(gripper_pos - block_pos) < 0.01:
                ac = self.goto_goal(block_pos, current_goal_block)

            else:
                ac = self.pickup_block(gripper_pos, block_pos, o[15])
            return ac

        gripper_pos = o[:3]
        block_pos0 = o[3:6]
        block_pos1 = o[6:9]
        relative_block_pos0 = o[9:12]
        relative_block_pos1 = o[12:15]
        current_goal_block0 = self.env.current_goal[-6:-3]
        current_goal_block1 = self.env.current_goal[-3:]

        # if the second block is placed in the right place
        if np.linalg.norm(block_pos1 - current_goal_block1) < 0.04:
            if self.full_space_as_goal:
                if np.linalg.norm(relative_block_pos1) < 0.04:
                    ac = np.array([0., 0., 1., 1.])
                else:
                    ac = self.goto_goal(gripper_pos, self.env.current_goal[:3], grip=1)
            else:
                ac = np.array([0., 0., 0., -1.])
        elif np.linalg.norm(block_pos0 - current_goal_block0) < 0.04:
            if np.linalg.norm(relative_block_pos0) < 0.1 and np.linalg.norm(relative_block_pos0[:2]) < 0.02:
                ac = np.array([0, 0, 1., 1.])
            else:
                ac  = deal_with_block(block_pos1, current_goal_block1, gripper_pos, relative_block_pos1)
        else:
            ac = deal_with_block(block_pos0, current_goal_block0, gripper_pos, relative_block_pos0)

        return ac


    def reset(self):
        pass


    def goto_block(self, cur_pos, block_pos):
        target_pos = block_pos + np.array([0, 0, 0.08])

        ac = clip_action((target_pos - cur_pos) / self.step_size)

        return np.concatenate([ac, np.array([1])])

    def pickup_block(self, cur_pos, block_pos, gripper_state):
        if np.linalg.norm(cur_pos - block_pos) < 0.01: #and gripper_state > 0.025: # TODO: need to adjust
            ac = np.array([0, 0, 0, -1.])
        else:
            ac = clip_action((block_pos - cur_pos) / self.step_size)
            ac = np.concatenate([ac, np.array([0.])])

        return ac


    def goto_goal(self, cur_pos, goal_pos, grip=-1):
        target_pos = goal_pos

        ac = clip_action((target_pos - cur_pos) / self.step_size)

        return np.concatenate([ac, np.array([grip])])



    def _random_action(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, 4))