from gym import Env, spaces
from gym.spaces import Tuple, Discrete, Box
import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Dict Of Actions
ALL_ACTIONS = {
    'U': 0,
    'D': 1,
    'C': 2,
    'L': 4,
    'R': 5,
    'UL': 6,
    'UR': 7,
    'DL': 8,
    'DR': 9
}

''' Gets action by name, and x y postion
    Return updated postion
'''
def move(action_name, x, y):
    assert action_name in ALL_ACTIONS.keys()

    if 'U' in action_name:
        x -= 1
    if 'D' in action_name:
        x += 1
    if 'L' in action_name:
        y -= 1
    if 'R' in action_name:
        y += 1

    return x, y


class WindyGridworld(Env):

    def __init__(self, height=20, width=10, rewards=[(1, 0, 5)], wind=True, start='random',
                 allowed_actions=['L', 'C', 'R'],
                 reward_terminates_episode = False):

        super(WindyGridworld,self).__init__()

        self.start = start
        self.height = height
        self.width = width
        self.rewards = rewards
        self.wind = wind
        self.claimed = np.ones(len(rewards))
        self.reward_terminates_episode = reward_terminates_episode

        assert all(
            map(
                lambda action: action in ALL_ACTIONS.keys(),
                allowed_actions
            )
        )

        self.actions = allowed_actions

        # Action Space attr required by gym
        self.action_space = Discrete(len(allowed_actions))

        self.canvas = None

        # observation space and shape required by gym
        self.observation_shape = (84, 84, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)



    @property
    def pos(self):
        if self.__pos is None:
            raise ValueError('environment reset needed')

        else:
            return self.__pos

    @property
    def opt_val(self):
        max_r = np.array(self.rewards)[:, 0].max()
        return max_r


    
    def set_pos(self, pos_x, pos_y):
        self.__pos = (pos_x, pos_y)
        self.canvas = visualize(self)
        
        
    '''Takes an integer input which represents an action.
    Returns observation image, Reward, and terminal state marker as required by gym
    '''    
    def step(self, action):

        x, y = self.pos

        # Check action 
        assert action in range(self.action_space.n)
        # Move
        x, y = move(self.actions[action], x, y)

        # Add wind
        if self.wind:
            x -= 1

        # Calculate Reward
        reward = 0
        for i, reward_spec in enumerate(self.rewards):
            value, x_, y_ = reward_spec
            if (x, y) == (x_, y_):
                reward += value
                self.claimed[i] = 0

        done = self.check_terminal_state(x, y)
        if self.reward_terminates_episode and reward > 0:
            done = True

        if sum(self.claimed) == 0:
            done = True
            #self.claimed = np.ones(len(self.rewards))

        # self.__pos = (x, y) if not done else None
        self.__pos = (x, y)

        self.canvas = visualize(self)

        return self.canvas, reward, done, {}
    
    '''Get player position and check to see if player has hit boundry of grid.
    '''
    def check_terminal_state(self, x, y):
        if x < 0: return True
        if y < 0: return True
        if x >= self.height: return True
        if y >= self.width: return True

        return False

    '''Reset the gym env. Return observation as required by gym
    '''
    def reset(self):

        if self.start == 'random':
            self.__pos = (self.height - 1, np.random.randint(0, self.width))

        else:
            try:
                x, y = self.start
                self.__pos = x, y
            except:
                raise KeyError()
                f'start parameter {self.start} not accepted'

        self.claimed = np.ones(len(self.rewards))
        self.canvas = visualize(self)

        return self.canvas

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(50)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

'''Get WindyGridWorld Object as input, 
return (84, 84, 3) image representation of gridworld'''
def visualize(env, black_bar=True):
    # Colors
    rgb_colors = {}
    for name, hex in matplotlib.colors.cnames.items():
        rgb_colors[name] = matplotlib.colors.to_rgb(hex)
    canvas = np.ones((env.height + 2, env.width + 2, 3)) * 255

    # Fill all but border
    canvas[1:-1, 1:-1, :] = 0

    # Draw reward
    for i, reward in enumerate(env.rewards):
        if env.claimed[i] != 0:
            value, x_, y_ = reward
            if value == 1:
                c = rgb_colors['green']

            else:
                c = rgb_colors['red']
            i = 0
            while i < 3:
                canvas[x_ + 1, y_ + 1, i] = c[i]
                i += 1

    # Draw player
    x, y = env.pos
    i = 0
    while i < 3:
        canvas[x + 1, y + 1, i] = rgb_colors['yellow'][i]
        i += 1

    # Add blackbar and make image square to keep aspect ratio of original grid world
    if black_bar:
        delta = canvas.shape[0] - canvas.shape[1]
        black_bar = np.zeros((env.height + 2, delta // 2))
        a = np.column_stack((black_bar, canvas[:, :, 0], black_bar))
        b = np.column_stack((black_bar, canvas[:, :, 1], black_bar))
        c = np.column_stack((black_bar, canvas[:, :, 2], black_bar))
        canvas = np.stack([a, b, c], axis=2)

    # Resive to 84 x 84
    r = cv2.resize(canvas[:, :, 0], (84, 84), interpolation=cv2.INTER_NEAREST)
    g = cv2.resize(canvas[:, :, 1], (84, 84), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(canvas[:, :, 2], (84, 84), interpolation=cv2.INTER_NEAREST)
    canvas = np.stack([r, g, b], axis=2)
    return canvas


if __name__ == '__main__':
    env = WindyGridworld(
        height=20,
        width=11,
        rewards=[[1, 0, 0], [2, 0, 10]],
        wind=True,
        allowed_actions=['L', 'R', 'C'],
        reward_terminates_episode=False
    )
    env.reset()
    steps = 100
    i = 0
    print(env.observation_space)
    while i < steps:

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        if done:
            env.reset()

        i += 1

    env.close()


