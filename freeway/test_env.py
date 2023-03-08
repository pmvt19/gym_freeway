import gym 
import numpy as np 
import torch
import cv2 as cv
import time 


UP = 1 
DOWN = 2

counter = 0

class Freeway(gym.Wrapper):
    def __init__(self):
        env = gym.make('ALE/Freeway-v5', full_action_space=False, mode=0, difficulty=1)
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

data = []
rewards = []
dones = []
terms = []
infos = []
max_steps = 1000

for i in range(max_steps):
    action = env.action_space.sample()
    action = 1

    next_state, reward, term, done, info = env.step(action)
    data.append((state, counter))

    if action == 1:
        counter += 1
    elif action == 2:
        counter = max(0, counter-1)

    dones.append(done)
    infos.append(info)
    terms.append(term)

    state = next_state 
    rewards.append(reward)

rewards = np.array(rewards)
dones = np.array(dones)
terms = np.array(terms)
print('Max: ', np.max(rewards), "Min: ", np.min(rewards))
print(np.unique(rewards))
print(np.unique(dones))
print(np.unique(terms))
# print(infos)

for frame, counter in data:

    if cv.waitKey(1) == 'q':
      break 
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    frame = cv.putText(frame, str(counter), (50, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv.LINE_AA)
    cv.imshow('newframe', frame)

    time.sleep(0.01)

cv.destroyAllWindows() 

