from rllab.misc.tensor_utils import stack_tensor_dict_list
import gym
import numpy as np

from sandbox.envs.pick_n_place.pick_n_place_gym import PnPEnv

"""Data generation for the case of a single block pick and place in Fetch Env"""


actions = []
observations = []
infos = []
successes = []
gs = []
ags = []

def noisify_action(env, action, exp_noise, exp_eps):
	if np.random.random() < exp_eps:
		return env.action_space.sample()
	return np.clip(np.array(action) + exp_noise * np.random.normal(size=4), -1, 1)



def collect_demos(num_demos, env, render=False, max_path_length=150, exp_noise=0., exp_eps=0.):
	global infos

	env.reset()
	print("Reset!")
	while len(actions) < num_demos:
		obs = env.reset()
		print("ITERATION NUMBER ", len(actions))
		goToGoal(env, obs, exp_noise=exp_noise, exp_eps=exp_eps, render=render, MAX_STEPS=max_path_length, )

	infos = stack_tensor_dict_list(infos)
	ret = dict(u=np.stack(actions), o=np.stack(observations), successes=np.stack(successes), g=np.stack(gs), ag=np.stack(ags))
	ret.update(infos)
	return ret


def goToGoal(env, lastObs, exp_noise, exp_eps, render=False, MAX_STEPS=150):

	episodeAcs = []
	episodeObs = []
	episodeInfo = []


	timeStep = 0 #count the total number of timesteps
	episodeObs.append(lastObs)

	def place_obj(block_num):
		nonlocal timeStep
		nonlocal lastObs

		goal = env.current_goal[block_num * 3: block_num * 3 + 3]
		objectPos = lastObs[3 + block_num * 3 : 6 + block_num * 3]
		object_rel_pos = lastObs[9 + block_num * 3 : 12 + block_num * 3]

		if np.linalg.norm(goal - objectPos) < 0.01:
			return

		object_oriented_goal = object_rel_pos.copy()
		object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object

		while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= MAX_STEPS:
			if render:
				env.render()
			action = [0, 0, 0, 0]
			object_oriented_goal = object_rel_pos.copy()
			object_oriented_goal[2] += 0.03

			for i in range(len(object_oriented_goal)):
				action[i] = object_oriented_goal[i]*6

			action[len(action)-1] = 0.05 #open

			action = noisify_action(env, action, exp_noise, exp_eps)
			obsDataNew, reward, done, info = env.step(action)
			timeStep += 1

			episodeAcs.append(action)
			episodeInfo.append(info)
			episodeObs.append(obsDataNew)

			objectPos = obsDataNew[3 + block_num * 3 : 6 + block_num * 3]
			object_rel_pos = obsDataNew[9 + block_num * 3 : 12 + block_num * 3]

		while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= MAX_STEPS:
			if render:
				env.render()
			action = [0, 0, 0, 0]
			for i in range(len(object_rel_pos)):
				action[i] = object_rel_pos[i]*6

			action[len(action)-1] = -0.005

			action = noisify_action(env, action, exp_noise, exp_eps)
			obsDataNew, reward, done, info = env.step(action)
			timeStep += 1

			episodeAcs.append(action)
			episodeInfo.append(info)
			episodeObs.append(obsDataNew)

			objectPos = obsDataNew[3 + block_num * 3 : 6 + block_num * 3]
			object_rel_pos = obsDataNew[9 + block_num * 3 : 12 + block_num * 3]


		while np.linalg.norm(goal - objectPos + np.array([0, 0, 0.1])) >= 0.01 and timeStep <= MAX_STEPS:
			if render:
				env.render()
			action = [0, 0, 0, 0]
			for i in range(len(goal - objectPos)):
				action[i] = (goal - objectPos + np.array([0, 0, 0.1]))[i]*6

			action[len(action)-1] = -0.005

			action = noisify_action(env, action, exp_noise, exp_eps)
			obsDataNew, reward, done, info = env.step(action)
			timeStep += 1

			episodeAcs.append(action)
			episodeInfo.append(info)
			episodeObs.append(obsDataNew)

			objectPos = obsDataNew[3 + block_num * 3 : 6 + block_num * 3]
			object_rel_pos = obsDataNew[9 + block_num * 3 : 12 + block_num * 3]


		while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= MAX_STEPS:
			if render:
				env.render()
			action = [0, 0, 0, 0]
			for i in range(len(goal - objectPos)):
				action[i] = (goal - objectPos)[i]*6

			action[len(action)-1] = -0.005

			action = noisify_action(env, action, exp_noise, exp_eps)
			obsDataNew, reward, done, info = env.step(action)
			timeStep += 1

			episodeAcs.append(action)
			episodeInfo.append(info)
			episodeObs.append(obsDataNew)

			objectPos = obsDataNew[3 + block_num * 3 : 6 + block_num * 3]
			object_rel_pos = obsDataNew[9 + block_num * 3 : 12 + block_num * 3]
		#
		# lastObs = obsDataNew

	place_obj(0)

	i = 0
	while i < 10 and timeStep < MAX_STEPS:
		if render:
			env.render()
		action = [0, 0, 0, 0]

		action[len(action) - 1] = 0.005

		if i > 5:
			action[2] = 1

		action = noisify_action(env, action, exp_noise, exp_eps)
		obsDataNew, reward, done, info = env.step(action)
		timeStep += 1

		episodeAcs.append(action)
		episodeInfo.append(info)
		episodeObs.append(obsDataNew)

		objectPos = obsDataNew[3:6]
		object_rel_pos = obsDataNew[9:12]

		i += 1



	place_obj(1)

	while True: #limit the number of timesteps in the episode to a fixed duration
		if render:
			env.render()
		action = [0, 0, 0, 0]
		action[len(action)-1] = -0.005 # keep the gripper closed

		# action = noisify_action(env, action, exp_noise, exp_eps)
		obsDataNew, reward, done, info = env.step(action)
		timeStep += 1

		episodeAcs.append(action)
		episodeInfo.append(info)
		episodeObs.append(obsDataNew)

		# objectPos = obsDataNew['observation'][3:6]
		# object_rel_pos = obsDataNew['observation'][6:9]

		if timeStep >= MAX_STEPS: break

	episodeInfo = stack_tensor_dict_list(episodeInfo)
	episodeInfo = {'info_%s' % k : v[:MAX_STEPS, None] if v.ndim==1 else v[:MAX_STEPS, ...] for k, v in episodeInfo.items()}

	episodeSuccesses = np.zeros(MAX_STEPS)
	episodeG = np.tile(np.array(env.current_goal), (MAX_STEPS, 1))
	episodeAg = np.stack([env.transform_to_goal_space(obs) for obs in episodeObs])

	actions.append(episodeAcs[:MAX_STEPS])
	observations.append(episodeObs[:MAX_STEPS+1])
	infos.append(episodeInfo)

	successes.append(episodeSuccesses)
	gs.append(episodeG[:MAX_STEPS])
	ags.append(episodeAg[:MAX_STEPS+1])


if __name__ == "__main__":
	collect_demos()
