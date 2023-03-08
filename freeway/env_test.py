import gym 
import numpy as np 
import torch
# from ale_py import ALEInterface
import cv2 as cv
import time 

# import sys
# np.set_printoptions(threshold=sys.maxsize)

UP = 1 
DOWN = 2

class Freeway(gym.Wrapper):
    def __init__(self):
        env = gym.make('ALE/Freeway-v5', full_action_space=False)
        super(Freeway, self).__init__(env)
        # self.example_obs = env.make_obs()

        self.observation_size = (210, 160, 3)
        self.action_size = 4
        self.preprocessors = []
        self.name = "freeway"

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, info

    def close(self):
        return self.env.close()

env = Freeway()

state, _ = env.reset()
print(state.shape)
# print(state)
print(type(env.action_space)) 
print(env.action_space) 

state, _ = env.reset() 

data = []
rewards = []
max_steps = 1000
# max_steps = 5
for i in range(max_steps):
    action = env.action_space.sample()
    # print(action)
    next_state, reward, done, _, _ = env.step(1)
    # state = state.transpose((2, 0, 1))
    data.append(state)
    state = next_state 
    rewards.append(reward)

# max_steps = 5
# for i in range(max_steps):
#     action = env.action_space.sample()
#     # print(action)
#     next_state, reward, done, _, _ = env.step(2)
#     # state = state.transpose((2, 0, 1))
#     data.append(state)
#     state = next_state 
#     rewards.append(reward)

# cv.imshow('frame', data[0])

rewards = np.array(rewards)
print('Max: ', np.max(rewards), "Min: ", np.min(rewards))

for frame in data:
    # print(frame.shape)

    if cv.waitKey(1) == 'q':
      break 
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow('newframe', frame)
#     # print("here")

    time.sleep(1)
# cv.waitKey(0)
cv.destroyAllWindows() 

