#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://stable-baselines3.readthedocs.io/en/master/guide/rl.html
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms


# # 1. Import dependencies

# In[2]:


import torch as torch
torch._C._cuda_getDeviceCount() > 0


# In[3]:


import os
import gym 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# # 2. Load and Test Environment

# In[4]:


#environment_name = "Pong-v0"
#rewards = [[1,0,5], [1,10,4], [1,4,6]]
#rewards = [[1,0,8], [1,10,3]]
#rewards = [[1,7,5],[1,0,4]]
rewards = [[1,0,5]]


# In[5]:


#env = gym.make(environment_name)
from GridEnv import WindyGridworld
env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )


# In[6]:


episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# 

# # 3. Train an RL Model

# In[7]:


from stable_baselines3.common.evaluation import evaluate_policy


# In[8]:


log_path = os.path.join('Training','Logs')


# In[9]:


from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


# In[10]:


save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')


# In[11]:


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

from stable_baselines3.common.atari_wrappers import WarpFrame

#env = WarpFrame(env)


# In[12]:


stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=save_path, 
                             verbose=1)


# In[13]:


# try 0.2 exploration period
model = DQN('CnnPolicy', env, exploration_fraction=0.4,
  exploration_final_eps=0.1,learning_starts=100000, verbose = 1, buffer_size = 50000,target_update_interval=1000, tensorboard_log=log_path)


# In[1]:


model.learn(total_timesteps=8000000,callback=eval_callback, tb_log_name='Env1_0.4exp_20kBuff_5m')


# In[ ]:


save_path = os.path.join('Training', 'Saved Models', 'DQN_model_Env1_0.4exp_20kBuff_5m')
#load_path = os.path.join('Training', 'Saved Models', 'DQN_model_GridWorld')
model.save(save_path)


# In[ ]:


evaluate_policy(model, env, n_eval_episodes=30, render=True)


# In[ ]:


env.close()


# In[ ]:




