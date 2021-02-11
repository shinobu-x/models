import numpy as np
import os
import pickle
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
# Generative Adversarial Networks
# https://arxiv.org/abs/1406.2661

def init_weight(network):
    for m in network.modules():
        classname = m.__class__.__name__
        weight_shape = list(m.weight.data.size())
        m.bias.data.fill_(0)
        if classname.find('Conv') != -1:
            l = np.prod(weight_shape[1: 4])
            r = np.prod(weight_shape[2: 4]) * weight_shape[0]
        elif classname.find('Linear') != -1:
            l = np.prod(weight_shape[1])
            r = np.prod(weight_spape[0])
        bound = np.sqrt(6.0 / (l + r))
        m.weight.data.uniform_(-bound, bound)

class Generator(nn.Module):
    def __init__(self, input_dim = 100, output_dim = 1, input_size = 32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128 * (self.input_size // 4) ** 2),
                nn.BatchNorm1d(128),
                nn.ReLU())
        self.deconv = nn.Sequential(
                nn.ConvTransposed2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Tanh())
        init_weight(self)

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, 128, (self.input_size // 4) ** 2)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 1, input_size = 32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.classifier = nn.Sequential(
                nn.Linear(128 * (self.input_size // 4) ** 2, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.output_dim),
                nn.Sigmoid())
        self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2))
        init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) ** 2)
        x = self.fc(x)
        return x

class GAN(object):
    def __init__(self, input_size, num_samples = 100, batch_size = 16,
            num_epochs = 10, z_dim = 62):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.input_size = input_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.z_dim = z_dim
        self.lr = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.transforms = transforms.Compose(
                [transforms.Resize(self.input_size), transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])])
        self.data_loader = DataLoader(datasets.MNIST("./", train = True,
            download = True, transform = self.transforms),
            batch_size = self.batch_size, shuffle = True)
        data = self.data_loader.__iter__().__next__()[0]
        self.G = Generator(input_dim = self.z_dim, output_dim = data.shape[1],
                input_size = self.input_size).to(self.device)
        self.D = Discriminator(input_dim = data.shape[1], output_dim = 1,
                input_size = self.input_size).to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = self.lr,
                betas = (self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = self.lr,
                betas = (self.beta1, self.beta2))
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.z = torch.rand((self.batch_size, self.z_dim)).to(self.device)

    def train(self):
        # min_G max_D V(D, G) =
        #   E_{x~p_{data}(x)}[log D(x)] + E_{z~p_z(z)}[log(1 - D(G(z))]
        # P(Y = y|x), if x comes from p_{data} (with y = 1), from p_g (with y =
        # 0) otherwise
        self.real = torch.ones(self.batch_size, 1).to(self.device)
        self.fake = torch.zeros(self.batch_size, 1).to(self.device)
        self.D.train()
        for epoch in range(self.epochs):
            self.G.train()
            # samples minibatch from data generating distribution
            for i, (x, _) in enumerate(self.data_loader):
                x.to(self.device)
                if i == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                # samples minibatch from noise prior p_g(z)
                z = torch.rand((self.batch_size, self.z_dim)).to(self.device)
                self.D_optimizer.zero_grad()
                # trains on real data
                # D(x^{(i)})
                D_real = self.D(x)
                D_real_loss = self.BCE_loss(D_real, self.real)
                # trains on fake data
                # D(G(z^{(i)}))
                D_fake = self.D(self.G(z))
                D_fake_loss = self.BCE_loss(D_fake, self.fake)
                D_loss = D_real_loss + D_fake_loss
                # updates the discriminator by ascending its stochastic gradient
                D_loss.backward()
                self.D_optimizer.step()
                self.G_optimizer.zero_grad()
                # trains on fake data
                # D(G(z^{(i)}))
                D_fake = self.D(self.G(z))
                G_loss = self.BCE_loss(D_fake, self.real)
                # updates the generator by descending its stochastic gradient
                G_loss.backward()
                self.G_optimizer.step()
