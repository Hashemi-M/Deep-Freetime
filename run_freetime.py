from sys import path as syspath
from os import path as ospath
syspath.append(ospath.join(ospath.expanduser("~"), '/data/hashemi/Freetime/Freetime-DQN/Deep-Freetime/SB3_f'))
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from SB3_f.sb3f import DQN

from SB3_f.sb3f.common.evaluation import evaluate_policy
from SB3_f.sb3f.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from Net_Qtable import Network_Qtable, Network_Ftable

def run_freetime(env, opt_val, env_num, steps, exp_frac, callback, exp_final):
    # rendering env
    episodes = 5
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

    # Define save paths
    save_path = os.path.join('Training', 'Saved Models')
    log_path = os.path.join('Training', 'Logs_F_with_Opt', 'Env '+str(env_num))
    log_name = 'Env{}_{}Steps_{}exp'.format(env_num, steps, exp_frac)


    # Create model
    opt_val = opt_val
    model = DQN('CnnPolicy', env, opt_val, exploration_fraction=exp_frac,
                exploration_final_eps=exp_final,
                learning_starts=100000, verbose=1,
                buffer_size=50000, target_update_interval=1000,
                tensorboard_log=log_path)

    # Get initial Q Table
    init_table = Network_Qtable(model, env)
    plt.imshow(init_table, cmap='jet')
    plt.title("Env " + str(env_num) + " Freetime Start")
    plt.colorbar()
    img_save_path =os.path.join('Q-Plot', 'Freetime', "Env " + str(env_num) + " Freetime Start")
    plt.savefig(img_save_path)
    plt.clf()


    # Check if callback is used, Train model
    if callback:
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
        eval_callback = EvalCallback(env,
                                     callback_on_new_best=stop_callback,
                                     eval_freq=10000,
                                     best_model_save_path=save_path,
                                     verbose=1)
        model.learn(total_timesteps=steps, callback=eval_callback, tb_log_name=log_name)

    else:
        model.learn(total_timesteps=steps, tb_log_name=log_name)

    # Save model
    save_path = os.path.join('Training', 'Saved Models', 'Env{}_{}Steps_{}exp'.format(env_num, steps, exp_frac))
    model.save(save_path)


    final_table = Network_Qtable(model, env)
    plt.imshow(final_table, cmap='jet')
    plt.title("Env " + str(env_num) + " Q-Table Freetime End")
    plt.colorbar()
    img_save_path = os.path.join('Q-Plot', "Env_" + str(env_num) + "_QT_Freetime_End")
    plt.savefig(img_save_path)
    plt.clf()


    # Get final Q table
    final_table = Network_Ftable(model, env)
    plt.imshow(final_table, cmap='jet')
    plt.title("Env " + str(env_num) + " F-Table Freetime End")
    plt.colorbar()
    img_save_path = os.path.join('Q-Plot', "Env_" + str(env_num) + "_FT_Freetime_End")
    plt.savefig(img_save_path)
    plt.clf()





