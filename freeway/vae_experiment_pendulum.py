import gym 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from vae_freeway_models import ConvVAE, ConvVAE_2, ConvVAE_3

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def vae_loss_fn(recon_x, x, mu, logvar, loss_function):
    RECON_LOSS = None 
    if loss_function == 'MSE':
        RECON_LOSS = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_function == 'BCE':
        RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction="sum")


    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON_LOSS + KLD, RECON_LOSS, KLD

class Pendulum(gym.Wrapper):
    def __init__(self):
        env = gym.make('Pendulum-v1', render_mode="rgb_array")
        super(Pendulum, self).__init__(env)

        self.observation_size = (210, 160, 3)
        self.action_size = 4
        self.preprocessors = []
        self.name = "Pendulum"
        self.my_state = None 

    def img_w_vel(self, obs, state):
        x, y, channels = obs.shape 
        # print(x, y, channels)
        new_obs = np.ones((x, y, channels+1))
        x, y, vel = state 
        

        new_obs[:, :, :3] = obs 
        new_obs[:, :, 3] = new_obs[:, :, 3] * vel 
        # print(new_obs)

        return new_obs


    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.my_state = obs
        obs = self.env.render()
        obs = self.img_w_vel(obs, self.my_state)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.my_state = obs
        obs = self.env.render()
        obs = self.img_w_vel(obs, self.my_state)
        return obs, reward, term, trunc, info
    
    def get_state(self):
        return self.my_state 

    def close(self):
        return self.env.close()

class StatesDataset(Dataset):

    def __init__(self, states):
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx])


def run_experiment(exp_num, network_type, features, loss_function, activation_function, epochs, batch_size, optimizer_function, likelihood_value, likelihood_modifier, max_total_steps, max_steps):
    env = Pendulum()
    env.reset()
    # img = env.render()
    # print(type(img), img.shape)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("Device:", DEVICE)

    #Setting Seeds 
    # SEED = 4891270
    # torch.manual_seed(SEED)
    # random.seed(SEED)
    # np.random.seed(SEED)
    # env.action_space.seed(SEED)
    
    # global using_conv
    MAX_TOTAL_STEPS = max_total_steps
    max_steps = max_steps
    batch_size = batch_size
    learning_rate = 1e-3

    activation = None 
    if activation_function == 'ReLU':
        activation = nn.ReLU()
    elif activation_function == 'Leaky ReLU':
        activation = nn.LeakyReLU()
    elif activation_function == 'Sigmoid':
        activation = nn.Sigmoid()
    elif activation_function == 'Tanh':
        activation = nn.Tanh()

    vae = ConvVAE_3(device=DEVICE)
    vae = vae.to(DEVICE)
    
    print(vae)

    vae_optimizer = None 
    if optimizer_function == 'Adam':
        vae_optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    elif optimizer_function == 'SGD':
        vae_optimizer = torch.optim.SGD(vae.parameters(), lr=learning_rate)

    total_steps = 0
    episodes = 0

    data = []
    while (total_steps < MAX_TOTAL_STEPS):
        episodes += 1

        state, _ = env.reset()

        for i in range(max_steps):
            action = env.action_space.sample() 
            next_state , _, done, _, _ = env.step(action) 
            state = state.transpose((2, 0, 1)) 
            print(state.shape)
            # return 
            data.append(state) 
            state = next_state 

        total_steps += max_steps 
        print(total_steps)

    dataset = StatesDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    start_time = time.time()
    epochs = 0 
    for epoch in range(epochs):
        running_loss = 0.0

        for idx, states in enumerate(dataloader, 0):

            states =  states / 255.0
            states = states.to(DEVICE)
            recon_states, mu, logvar = vae(states)

            loss, recon_loss, kld = vae_loss_fn(recon_states, states, mu, logvar, loss_function)
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()

        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)

        print(to_print, end="\n")
    print(f"**Finished VAE Training: total time: {time.time()-start_time:3.2f}**")

    test_data = [] 
    total_steps = 0
    episodes = 0
    while (total_steps < 500):
        episodes += 1

        state, _ = env.reset()

        for i in range(max_steps):
            action = env.action_space.sample()
            next_state , _, done, _, _ = env.step(action)
            state = state.transpose((2, 0, 1))
            x, y, vel = env.get_state()
            rho, theta = cart2pol(x, y)

            test_data.append((state, theta))
            state = next_state

        total_steps += max_steps 
        print(total_steps)

    thetas = []
    likelihoods = [] 

    with torch.no_grad():
        for state, theta in test_data:
            state = torch.tensor(state)
            state = state / 255
            state = state.to(DEVICE)
            state = state[None, :, :, :]
            recon_state, mu, logvar = vae(state)
            loss, recon_loss, kld = vae_loss_fn(recon_state, state, mu, logvar, loss_function) 

            likelihood_val = None 

            if likelihood_value == 'loss':
                likelihood_val = loss 
            elif likelihood_value == 'kld':
                likelihood_val = kld 
            elif likelihood_value == 'recon_loss':
                likelihood_val = recon_loss 
            likelihood_val = float(likelihood_val)
            thetas.append(theta)
            if likelihood_modifier == 'negative':
                likelihoods.append(-likelihood_val)
            elif likelihood_modifier == 'inverse':
                epsilon = 0.1**5
                likelihoods.append((1 / (likelihood_val + epsilon)))

    plt.figure()
    thetas = np.array(thetas)
    likelihoods = np.array(likelihoods)
    print(type(thetas), type(likelihoods), thetas.dtype, likelihoods.dtype)
    plt.scatter(thetas, likelihoods)
    plt.savefig('pendulum_likelihoods.png')
