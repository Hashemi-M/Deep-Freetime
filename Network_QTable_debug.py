#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gym 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[2]:


from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"), '/data/hashemi/Freetime/Freetime-DQN/Deep-Freetime/SB3_f'))


# In[ ]:





# In[3]:


# import os
# import gym 
# import matplotlib.pyplot as plt
# import numpy as np 
# from SB3_f.sb3f import DQN
# from SB3_f.sb3f.common.vec_env import DummyVecEnv
# from SB3_f.sb3f.common.evaluation import evaluate_policy
# from SB3_f.sb3f.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


# In[4]:


#environment_name = "Pong-v0"

#rewards = [[1,1,6],[1,3,4]]
rewards = [[1,0,5]]
#rewards = [[1,1,6],[1,3,4]]


# In[5]:


''''
#env = gym.make(environment_name)
from GridEnv import WindyGridworld
env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )'''

from GridEnv import WindyGridworld
env = env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )

print(env.rewards)


# In[6]:


log_path = os.path.join('Training', 'Logs')
#model_name = 'DQN_model_Env1_0.4exp_20kBuff_5m'
model_name = 'Env1_Baseline_0.2exp_20kBuff_8m'
load_path = os.path.join('Training', 'Saved Models',model_name)
model = DQN.load(load_path, env = env)


# In[ ]:





# In[7]:


evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()


# In[8]:


env.close()


# In[9]:


from GridEnv import *


# In[18]:


# Genereate Image for each state, maybe dict{state: img}

env_w = env.width
env_l = env.height

w = 0
# Loop width of env
state_img_dict = dict()
while w < env_w:
    l = 0
    # Loop height of env
    while l< env_l:

        env.set_pos(l,w)
        img = visualize(env)
        state_img_dict[(l,w)] = img
        
        l+=1
        
    w += 1

print(state_img_dict.keys())
# loop the dict and add the value for each image{}

# Render the state, value 


# In[11]:


model.predict(state_img_dict[(0, 4)],deterministic=True)

import matplotlib.pyplot as plt
import torch as th 
from stable_baselines3.common.utils import obs_as_tensor

device = th.device('cuda:0')

print(device)


obs = state_img_dict[(0, 4)]
print(obs.shape)

observation = np.transpose(obs,(2,0,1))
observation = observation[np.newaxis,...]
print(observation.shape)
plt.imshow(observation[0,0,:,:])
#plt.imshow(obs[:,:,0])

observation = obs_as_tensor(observation, device)
with th.no_grad():
    q_values = model.q_net(observation)

q_values.cpu().detach().numpy()


# In[12]:


# loop the dict and add the value for each image{}
state_img_dict[(0, 0)]

img = np.ones((20,11))

for key in state_img_dict.keys():
    state = key
    obs = state_img_dict[key]
                
    observation = np.transpose(obs,(2,0,1))
    observation = observation[np.newaxis,...]
    observation = obs_as_tensor(observation, device)
    with th.no_grad():
        q_values = model.q_net(observation)
        
    max_val = np.max(q_values.cpu().detach().numpy())

    
    img[state[0],state[1]] = max_val


# In[13]:


#img


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


plt.imshow(img, cmap='jet')
plt.title("Env1 Baseline")
plt.colorbar()
plt.show()


# In[16]:



plt.savefig("Env2-Qval.png")


# In[17]:


import numpy as np

def Network_Qtable(model,env):
    
    # Genereate Image for each state, maybe dict{state: img}
    env_w = env.width
    env_l = env.height

    w = 0
    # Loop width of env
    state_img_dict = dict()
    while w < env_w:
        l = 0
        # Loop height of env
        while l< env_l:
            env.reset()
            env.set_pos(l,w)
            img = visualize(env)
            state_img_dict[(l,w)] = img

            l+=1

        w += 1
    
    img = np.ones((20,11))

    for key in state_img_dict.keys():
        state = key
        obs = state_img_dict[key]

        observation = np.transpose(obs,(2,0,1))
        observation = observation[np.newaxis,...]
        observation = obs_as_tensor(observation, device)
        with th.no_grad():
            q_values = model.q_net(observation)

        max_val = np.max(q_values.cpu().detach().numpy())


        img[state[0],state[1]] = max_val
    
    return img

