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
    def __init__(self, goal_weight=1., terminal_eps=0.05):
        Serializable.quick_init(self, locals())

        env = gym.envs.make('FetchPickAndPlace-v1')
        self.env = env
        self.env_id = env.spec.id

        self._observation_space = convert_gym_space(env.observation_space.spaces['observation'])
        self._action_space = convert_gym_space(env.action_space)

        self._current_goal = None

        self.goal_weight = goal_weight
        self.terminal_eps = terminal_eps


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
        self._current_goal = d['desired_goal']
        return self._transform_obs(d['observation'])

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        info['obs2goal'] = next_obs['achieved_goal']
        info['goal_reached'] = info['is_success']
        info['distance'] = np.linalg.norm(self.current_goal - info['obs2goal'])
        return Step(self._transform_obs(next_obs['observation']), reward, done, **info)


    def transform_to_goal_space(self, obs):
        return obs[3:6]

    def render(self):
        self.env.render()


    def _transform_obs(self, obs):
        # TODO: for now is the identity transformation
        return obs[:10]
