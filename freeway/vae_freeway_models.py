import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchsummary import summary


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def __init__(self, size=1024):
        super(UnFlatten, self).__init__()
        self.size = size 

    def forward(self, input):
        # return input.view(input.size(0), self.size, 4, 3)
        return input.view(input.size(0), 64, 23, 17)

class ConvVAE(nn.Module):
    def __init__(self, device='cpu'):
        super(ConvVAE, self).__init__()
        self.device = device 
        image_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            Flatten()
        )

        h_dim = 25024
        z_dim = 32 
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(size=h_dim),
            # nn.ConvTranspose2d(h_dim, 64, kernel_size=9, stride=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2, padding=(0, 1)),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print(h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        # print(z.size())
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print(z.size())
        z = self.decode(z)
        return z, mu, logvar


class UnFlatten_2(nn.Module):
    def __init__(self, size=1024):
        super(UnFlatten_2, self).__init__()
        self.size = size 

    def forward(self, input):
        # return input.view(input.size(0), self.size, 4, 3)
        return input.view(input.size(0), 64, 59, 59)

class ConvVAE_2(nn.Module):
    def __init__(self, device='cpu'):
        super(ConvVAE_2, self).__init__()
        self.device = device 
        image_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            Flatten()
        )

        h_dim = 222784
        z_dim = 8 
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten_2(size=h_dim),
            # nn.ConvTranspose2d(h_dim, 64, kernel_size=9, stride=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print(h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        # print(z.size())
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print(z.size())
        z = self.decode(z)
        return z, mu, logvar

class UnFlatten_3(nn.Module):
    def __init__(self, size=1024):
        super(UnFlatten_3, self).__init__()
        self.size = size 

    def forward(self, input):
        # return input.view(input.size(0), self.size, 4, 3)
        return input.view(input.size(0), 16, 59, 59)

class ConvVAE_3(nn.Module):
    def __init__(self, device='cpu'):
        super(ConvVAE_3, self).__init__()
        self.device = device 
        image_channels = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            Flatten()
        )

        h_dim = 55696
        z_dim = 8 
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten_3(size=h_dim),
            # nn.ConvTranspose2d(h_dim, 64, kernel_size=9, stride=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print(h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        # print(z.size())
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # print(z.size())
        z = self.decode(z)
        return z, mu, logvar