import numpy as np
from GridEnv import *
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
device = th.device('cuda:0')

def Network_Qtable(model, env):

    device = th.device('cuda:0')

    # Genereate Image for each state, maybe dict{state: img}
    env_w = 11
    env_l = 20

    w = 0
    # Loop width of env
    state_img_dict = dict()
    while w < env_w:
        l = 0
        # Loop height of env
        while l < env_l:

            env.set_pos(l, w)
            img = visualize(env)
            state_img_dict[(l, w)] = img

            l += 1

        w += 1

    img = np.ones((20, 11))

    for key in state_img_dict.keys():
        state = key
        obs = state_img_dict[key]

        observation = np.transpose(obs, (2, 0, 1))
        observation = observation[np.newaxis, ...]
        observation = obs_as_tensor(observation, device)
        with th.no_grad():
            q_values = model.q_net(observation)

        max_val = np.max(q_values.cpu().detach().numpy())

        img[state[0], state[1]] = max_val

    return img


def Network_Ftable(model, env, type="max"):
    device = th.device('cuda:0')

    # Genereate Image for each state, maybe dict{state: img}
    env_w = 11
    env_l = 20

    w = 0
    # Loop width of env
    state_img_dict = dict()
    while w < env_w:
        l = 0
        # Loop height of env
        while l < env_l:
            env.set_pos(l, w)
            img = visualize(env)
            state_img_dict[(l, w)] = img

            l += 1

        w += 1

    img = np.ones((20, 11))

    for key in state_img_dict.keys():
        state = key
        obs = state_img_dict[key]

        observation = np.transpose(obs, (2, 0, 1))
        observation = observation[np.newaxis, ...]
        observation = obs_as_tensor(observation, device)
        with th.no_grad():
            q_values = model.f_net(observation)

        if type == 'max':
            max_val = np.max(q_values.cpu().detach().numpy())
        else:
            max_val = np.min(q_values.cpu().detach().numpy())

        img[state[0], state[1]] = max_val

    return img
