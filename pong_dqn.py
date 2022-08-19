import gym
import time
import torch
print(torch.cuda.get_device_name(0))
torch.device('cuda')

env = gym.make("Pong-v4")
env.reset()

for step in range(200):
    env.render()
    time.sleep(0.01)
    env.step(env.action_space.sample())

env.close()


