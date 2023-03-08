import time 
import gym
import matplotlib.pyplot as plt 
import cv2 
import numpy as np 

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

env = gym.make('Pendulum-v1', render_mode="rgb_array")

# Number of steps you run the agent for 
num_steps = 1000

obs = env.reset()

images = []

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    # action = 20
    obs, reward, done, term, info = env.step(action)
    
    # Render the env
    x, y, vel = obs 
    rho, theta = cart2pol(x, y)
    img = env.render()
    img = cv2.putText(img, str(theta), org, font, fontScale, color, thickness, cv2.LINE_AA)
    images.append(img)
    # plt.imshow(img)
    # plt.show()


    # Wait a bit before the next frame unless you want to see a crazy fast video
    # time.sleep(0.0001)
    
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()

for img in images:
    cv2.imshow('frame', img)
    time.sleep(0.1)

    if cv2.waitKey(1) == 'q':
        break 

cv2.destroyAllWindows()


