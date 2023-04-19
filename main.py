from GridEnv import WindyGridworld
from run_freetime import run_freetime

def gen_env(env_num):
    if env_num == 1:
        rewards = [[1, 0, 5]]
    elif env_num == 2:
        rewards = [[1, 1, 6], [1, 3, 4]]
    elif env_num == 3:
        rewards = [[2, 0, 10], [1, 0, 0]]
        terminates = True
    if env_num == 4:
        rewards = [[1, 10, 5]]
        env = WindyGridworld(
            height=20,
            width=11,
            rewards=rewards,
            wind=False,
            allowed_actions=['L', 'R', 'U', 'D'],
            reward_terminates_episode=True
        )
    else:
        env = WindyGridworld(
        height=20,
        width=11,
        rewards=rewards,
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
        )
    return env

def main(baseline = False, env_num=1, steps=1000000, exp_frac=0.3, callback=True,exp_final = 0.1, opt_val=1, runs=1):
    # Create Env
    env = gen_env(env_num)
    # run optimist
    run = 0
    while run < runs:
        run_freetime(env, opt_val, env_num, steps, exp_frac, callback, exp_final)



    if baseline:
        pass
        #run_baseline(env, env_num, steps, exp_frac, callback, exp_final)


if __name__ == '__main__':


    env_num = 1
    baseline = False
    steps = 1000000
    exp_frac = 0.8
    callback = True
    exp_final = 0.1
    opt_val = 1
    runs = 3
    main(baseline, env_num, steps, exp_frac, callback, exp_final, opt_val, runs)