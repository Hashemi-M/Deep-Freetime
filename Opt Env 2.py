


from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"), '/data/hashemi/Freetime/Freetime-DQN/Deep-Freetime/SB3_f'))


import torch as torch

import os
import gym 
import matplotlib.pyplot as plt
import numpy as np 
from SB3_f.sb3f import DQN
from SB3_f.sb3f.common.vec_env import DummyVecEnv
from SB3_f.sb3f.common.evaluation import evaluate_policy
from SB3_f.sb3f.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


rewards = [[1,1,6],[1,3,4]]
from GridEnv import WindyGridworld
env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )


# In[5]:


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


# In[6]:


save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs Bias Only', 'Env 2')


# In[7]:


stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=save_path, 
                             verbose=1)



opt_val = env.opt_val
model = DQN('CnnPolicy', env, opt_val, exploration_fraction=0.2,
  exploration_final_eps=0.1,learning_starts=1000, verbose = 1, buffer_size = 50000,target_update_interval=1000, tensorboard_log=log_path)



model.learn(total_timesteps=1000000,callback=eval_callback, tb_log_name='Env2_0.2exp_1M')




save_path = os.path.join('Training', 'Saved Models', 'Env2_Opt_BaisOnly_1M')
model.save(save_path)

del(model)

