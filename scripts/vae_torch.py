 """PyTorch implementation of MMD-VAE described in Info-VAE paper."""
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np

class AEData(torch.utils.data.Dataset):
    def __init__(self,x):
        super(AEData, self).__init__()
        self.x = x
        self.y = x


    def __len__(self):
        return self.y.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MMD_VAE(nn.Module):
    def __init__(self, size=None, num_latent=1, depth=1):
        super(MMD_VAE, self).__init__()
        encode_list = [nn.Linear(size, 128*depth)]
        for i in range(depth-1):
            encode_list.append(nn.Linear(128*(depth - i), 128*(depth - i - 1)))
        encode_list.append(nn.Linear(128,num_latent))
        self.encode_net = nn.ModuleList(encode_list)
        decode_list = [nn.Linear(num_latent,128)]
        for i in range(depth-1):
            decode_list.append(nn.Linear(128*(i+1), 128 * (i+2)))
        decode_list.append(nn.Linear(128*depth,size))
        self.decode_net = nn.ModuleList(decode_list)


    def encode(self,x):
        for i in range(len(self.encode_net) - 1):
            x = F.relu(self.encode_net[i](x))
        x = self.encode_net[-1](x)
        return x

    def decode(self,z):
        for i in range(len(self.decode_net) - 1):
            z = F.relu(self.decode_net[i](z))
        z = self.decode_net[-1](z)
        return z


    def forward(self,x):
        return self.decode(self.encode(x))


def compute_kernel(x,y):
    x_size, dim = x.size()
    y_size = y.size()[0]
    tiled_x = x.reshape(x_size,1,dim).repeat([1,y_size,1])
    tiled_y = y.reshape(1,y_size,dim).repeat([x_size,1,1])
    if torch.cuda.is_available():
        tiled_x = tiled_x.cuda()
        tiled_y = tiled_y.cuda()
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,2) / float(dim))


def compute_mmd(x,y):
    x_kernel = compute_kernel(x,x)
    y_kernel = compute_kernel(y,y)
    xy_kernel = compute_kernel(x,y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)


def loss(train_z,output,x):
    samples = torch.randn([200,train_z.size()[1]])
    loss_mmd = compute_mmd(samples, train_z)
    loss_nll = torch.mean((output - x)**2)
    return loss_mmd + loss_nll


def train_mmd_vae(genotype, params, verbose=False, plot_loss=False, save_loss=False):
    model = nn.DataParallel(MMD_VAE(**params))
    if torch.cuda.is_available():
        model = model.cuda()
        genotype = torch.Tensor(genotype).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    traindata = AEData(genotype)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=True)
    tol = 1e10
    i = 0
    prev_loss = 1e10
    losses = []
    while(i < 100): #num epochs
        i += 1
        for data in trainloader:
            optimizer.zero_grad()
            output = model(genotype)
            loss_fn = loss(model.module.encode(genotype),output, genotype)
            tol = abs(prev_loss - loss_fn)
            prev_loss = loss_fn
            loss_fn.backward()
            optimizer.step()
        if plot_loss:
            losses.append(loss_fn)
            plt.clf()
            plt.plot(np.arange(i),np.array(losses))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.draw()
            plt.pause(0.001)
        if verbose:
            print("Loss at epoch {}: {}".format(i,loss_fn))
    if save_loss:
        plt.plot(np.arange(i),np.array(losses))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss_fig.png")
    del genotype
    return model.cpu().module
