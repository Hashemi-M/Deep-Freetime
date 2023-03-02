#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://stable-baselines3.readthedocs.io/en/master/guide/rl.html
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms


# # 1. Import dependencies

# In[2]:


from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"), '/data/hashemi/Freetime/Freetime-DQN/Deep-Freetime/SB3_f'))


# In[3]:


import os
import gym 
import matplotlib.pyplot as plt
import numpy as np 
from SB3_f.sb3f import DQN
from SB3_f.sb3f.common.vec_env import DummyVecEnv
from SB3_f.sb3f.common.evaluation import evaluate_policy
from SB3_f.sb3f.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from SB3_f.sb3f.common.atari_wrappers import AtariWrapper


# # 2. Load and Test Environment

# In[4]:


environment_name = "Pong-v4"


# In[5]:


env = gym.make(environment_name)


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


log_path = os.path.join('Training','Logs Opt Atari','Pong')
#training_log_path = os.path.join(log_path, 'DQN_Pong')


# In[ ]:





# In[8]:


save_path = os.path.join('Training', 'Saved Models')


# In[9]:


env = gym.make(environment_name)
env = AtariWrapper(env)


# In[10]:


stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=save_path, 
                             verbose=1)


# In[11]:


# Pong Reward of -1 or +1 so Opt is +1
opt_val = 1

model_pong = DQN('CnnPolicy', env, opt_val, verbose = 1,
            buffer_size = 100000,
            learning_rate = 0.0001, 
            batch_size = 32,
            learning_starts = 100000,
            target_update_interval = 1000,
            train_freq = 4,
            gradient_steps =  1,
            exploration_fraction = 0.1,
            exploration_final_eps = 0.01,
            optimize_memory_usage = False,
            tensorboard_log=log_path)


# In[12]:


model_pong.learn(total_timesteps=10000000, callback=eval_callback,tb_log_name='Pong_Opt_10x0.1')


# In[34]:


save_path = os.path.join('Training', 'Saved Models','Opt_Pong_10M')
model_pong.save(save_path)


# In[36]:


evaluate_policy(model_pong, env, n_eval_episodes=10, render=True)


# In[21]:


env.close()


# In[39]:





# In[ ]:




