import os
from gym import utils
from sandbox.envs.pick_n_place.fetch_env_twoobj import FetchEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asset', 'fetch', 'pick_and_place_twoobj.xml')


class FetchPickAndPlaceEnv(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', stacking=False, first_in_place=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, stacking=stacking, first_in_place=first_in_place)
        utils.EzPickle.__init__(self)