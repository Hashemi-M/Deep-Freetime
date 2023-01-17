from sys import path as syspath
from os import path as ospath

import numpy as np
import torch as th
syspath.append(ospath.join(ospath.expanduser("~"), '/data/hashemi/Freetime/Freetime-DQN/Deep-Freetime/SB3_f'))

import gym
from SB3_f.sb3f import DQN
from SB3_f.sb3f.common.vec_env import DummyVecEnv
from SB3_f.sb3f.common.evaluation import evaluate_policy
from GridEnv import *
import torch as th
from SB3_f.sb3f.common.utils import obs_as_tensor


rewards = [[3, 0, 5]]

env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )

print(env.opt_val)
opt_val = env.opt_val
model = DQN('CnnPolicy', env, opt_val, verbose = 1)

#evaluate_policy(model, env = model.env , n_eval_episodes=10, render=True)
env.close()

# Genereate Image for each state, maybe dict{state: img}
env_w = env.width
env_l = env.height

w = 0
# Loop width of env
state_img_dict = dict()

while w < env_w:
        l = 0
        # Loop height of env
        while l < env_l:
                env.reset()
                env.set_pos(l, w)
                img = visualize(env)
                state_img_dict[(l, w)] = img

                l += 1

        w += 1

print(state_img_dict.keys())

# loop the dict and add the value for each image{}
model.predict(state_img_dict[(0, 4)], deterministic=True)


device = th.device('cuda:0')
print(device)


img_max = np.ones((20,11))
img_min = np.ones((20,11))
img_mean = np.ones((20,11))

for key in state_img_dict.keys():
        state = key

        obs = state_img_dict[key]

        observation = np.transpose(obs, (2, 0, 1))
        observation = observation[np.newaxis, ...]
        observation = obs_as_tensor(observation, device)
        with th.no_grad():
                q_values = model.q_net(observation)

        max_val = np.max(q_values.cpu().detach().numpy())
        min_val = np.min(q_values.cpu().detach().numpy())
        mean_val = np.mean(q_values.cpu().detach().numpy())

        img_max[state[0], state[1]] = max_val
        img_min[state[0], state[1]] = min_val
        img_mean[state[0], state[1]] = mean_val

#plt.imshow(img_mean, cmap='jet')
#plt.title("mean Q val")
#plt.colorbar()
#plt.show()

print(np.min(img_min), np.max(img_max))




