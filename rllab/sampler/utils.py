import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, init_state=None, no_action = False, using_gym=False, noise=0, o=None, plan=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []
    # no_action = True
    if o is None:
        if init_state is not None:
            o = env.reset(init_state)
        else:
            o = env.reset()
        agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        if not using_gym:
            a, agent_info = agent.get_action(o)
        else:
            if hasattr(agent, 'relative_goals') and agent.relative_goals:
                ag = env_info['xy_pos'] if len(env_infos) > 0 else env.init_goal_obs
                goal = plan(ag, env.current_goal) if plan is not None else env.current_goal
                a = agent.get_actions([o], ag, goal, noise_eps=noise)
                agent_infos = None
            else:
                a = agent.get_actions([o], env.transform_to_goal_space(o), env.current_goal, noise_eps=noise)
                # a = agent.get_actions([o], np.zeros_like(env.current_goal), np.zeros_like(env.current_goal), noise_eps=noise)
                agent_infos = None

        if no_action:
            a = np.zeros_like(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        if agent_infos is not None:
            agent_infos.append(agent_info)
        env_infos.append(env_info)
        dones.append(d)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=False)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        # agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos) if agent_infos is not None else None,
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        dones=np.asarray(dones),
        last_obs=o,
    )
