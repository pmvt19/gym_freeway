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

from vae_freeway_models import ConvVAE

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def vae_loss_fn(recon_x, x, mu, logvar, loss_function):
    RECON_LOSS = None 
    if loss_function == 'MSE':
        RECON_LOSS = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_function == 'BCE':
        RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction="sum")


    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON_LOSS + KLD, RECON_LOSS, KLD

class Freeway(gym.Wrapper):
    def __init__(self):
        env = gym.make('ALE/Freeway-v5', full_action_space=False)
        super(Freeway, self).__init__(env)

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

class StatesDataset(Dataset):

    def __init__(self, states):
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx])


def run_experiment(exp_num, network_type, features, loss_function, activation_function, epochs, batch_size, optimizer_function, likelihood_value, likelihood_modifier, max_total_steps, max_steps):
    env = Freeway()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("Device:", DEVICE)

    #Setting Seeds 
    SEED = 4891270
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    env.action_space.seed(SEED)
    
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

    vae = ConvVAE(device=DEVICE)
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
            data.append(state)
            state = next_state 

        total_steps += max_steps 

    dataset = StatesDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0

        for idx, states in enumerate(dataloader, 0):

            states =  states / 255.0
            # print(states.shape)
            recon_states, mu, logvar = vae(states)

            loss, recon_loss, kld = vae_loss_fn(recon_states, states, mu, logvar, loss_function) #TODO: should be next_images not images
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()

        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)

        print(to_print, end="\n")
    print(f"**Finished VAE Training: total time: {time.time()-start_time:3.2f}**")