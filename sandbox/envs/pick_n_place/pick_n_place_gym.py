import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
import numpy as np

import gym


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError


class PnPEnv(Env, Serializable):
    def __init__(self, goal_weight=1., terminal_eps=0.05, full_space_as_goal=False, feasible_hand=True, two_obj=False,
                 first_in_place=False, stacking=False, fix_goal=False):
        Serializable.quick_init(self, locals())

        # env = gym.envs.make('FetchPickAndPlace-v1')
        if not two_obj:
            from sandbox.envs.pick_n_place.pick_and_place_ontable import FetchPickAndPlaceEnv
            env = FetchPickAndPlaceEnv()
        else:
            from sandbox.envs.pick_n_place.pick_and_place_twoobj import FetchPickAndPlaceEnv
            env = FetchPickAndPlaceEnv(stacking=stacking, first_in_place=first_in_place)

        env.unwrapped.spec=self
        self.env = env
        # self.env_id = env.spec.id

        self._observation_space = convert_gym_space(env.observation_space.spaces['observation'])
        self._action_space = convert_gym_space(env.action_space)

        self._current_goal = None

        self.goal_weight = goal_weight
        self.terminal_eps = terminal_eps
        self.full_space_as_goal = full_space_as_goal
        self.two_obj = two_obj

        self.feasible_hand = feasible_hand # if the position of the hand is always feasible to achieve

        self.fix_goal = fix_goal
        if fix_goal:
            self.fixed_goal = np.array([1.48673746, 0.69548325, 0.6])



    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def current_goal(self):
        return self._current_goal

    def reset(self):
        d = self.env.reset()
        self.update_goal(d=d)
        return self._transform_obs(d['observation'])

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self._transform_obs(next_obs['observation'])
        info['obs2goal'] = self.transform_to_goal_space(next_obs)
        info['distance'] = np.linalg.norm(self.current_goal - info['obs2goal'])
        if self.full_space_as_goal:
            info['block_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[3:6])
            info['hand_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[0:3])
        info['goal_reached'] = info['distance'] < self.terminal_eps
        return Step(next_obs, reward, done and info['goal_reached'], **info)

    def get_current_obs(self):
        return self._transform_obs(self.env._get_obs()['observation'])

    def transform_to_goal_space(self, obs):
        if not self.full_space_as_goal:
            ret = np.array(obs[3:6])
        else:
            ret = np.array(obs[:6])
        if self.two_obj:
            ret = np.concatenate([ret, obs[6:9]])
        return ret

    def render(self):
        self.env.render()


    def _transform_obs(self, obs):
        if self.two_obj:
            return obs[:16]
        else:
            return obs[:10]

    def sample_hand_pos(self, block_pos):
        if block_pos[2] == self.env.height_offset or not self.feasible_hand:
            xy = self.env.initial_gripper_xpos[:2] + np.random.uniform(-0.15, 0.15, size=2)
            z = np.random.uniform(self.env.height_offset, self.env.height_offset + 0.3)
            return np.concatenate([xy, [z]])
        else:
            return block_pos


    def update_goal(self, d=None):
        if self.get_current_obs()[5] < self.env.height_offset or \
                np.any(self.get_current_obs()[3:5] > self.env.initial_gripper_xpos[:2] + 0.15) or \
                np.any(self.get_current_obs()[3:5] < self.env.initial_gripper_xpos[:2] - 0.15):
            self.env._reset_sim()
        if self.fix_goal:
            self._current_goal = self.fixed_goal
        else:
            if d is not None:
                self._current_goal = d['desired_goal']
            else:
                self._current_goal = self.env.goal = np.copy(self.env._sample_goal())
            if self.full_space_as_goal:
                self._current_goal = np.concatenate([self.sample_hand_pos(self._current_goal), self._current_goal])

    def set_feasible_hand(self, bool):
        self.feasible_hand = bool
