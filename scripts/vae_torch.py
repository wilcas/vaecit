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


    def __len__(self):
        return self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index,:]


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


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def loss_mmd_nll(train_z,output,x):
    if torch.cuda.is_available():
        samples = torch.randn([200,train_z.size()[1]]).cuda()
    else:
        samples = torch.randn([200,train_z.size()[1]]).cuda()
    loss_mmd = compute_mmd(samples, train_z)
    loss_nll = (output - x).pow(2).mean()
    return loss_mmd, loss_nll


def train_mmd_vae(genotype, params, verbose=False, plot_loss=False, save_loss=False):
    model = nn.DataParallel(MMD_VAE(**params))
    if torch.cuda.is_available():
        model = model.cuda()
        genotype = torch.Tensor(genotype).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    traindata = AEData(genotype)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=True)
    tol = 1e10
    i = 0
    prev_loss = 1e10
    losses = []
    loss_value = None
    while(i < 50): #num epochs
        i += 1
        for (j,gen_batch) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(gen_batch)
            loss_mmd,loss_nll = loss_mmd_nll(model.module.encode(gen_batch),output,gen_batch)
            loss = loss_mmd + loss_nll
            loss.backward()
            optimizer.step()
            if verbose:
                print("Loss at batch {}, epoch {}: nll: {}, mmd: {}".format(j,i,loss_nll, loss_mmd))
        output = model(genotype.detach()).detach()
        loss_mmd,loss_nll = loss_mmd_nll(model.module.encode(genotype.detach()).detach(),output.detach(),genotype.detach())
        loss = loss_mmd.detach() + loss_nll.detach()
        losses.append(loss.item())
        if plot_loss:
            plt.clf()
            plt.plot(np.arange(i),np.array(losses))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.draw()
            plt.pause(0.001)
        if verbose:
            print("Loss at epoch {}: {}".format(i,loss.item()))
    if save_loss:
        plt.plot(np.arange(i),np.array(losses))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(save_loss)
        plt.close()
    del genotype
    return model.cpu().module
