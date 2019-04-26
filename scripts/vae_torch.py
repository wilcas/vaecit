"""PyTorch implementation of MMD-VAE described in Info-VAE paper."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MMD_VAE(nn.Module):
    def __init__(self, size=None, num_latent=1, depth=1):
        self.encode_net = [nn.Linear(input_size,128*depth)]
        for i in range(depth):
            self.encode_net.append(nn.Linear(128*(depth - i - 1), 128 * (depth - i)))
        self.encode_net.append(nn.Linear(128,num_latent))
        self.decode_net = [nn.Linear(num_latent,128)]
        for i in range(depth-1):
            self.decode_net.append(nn.Linear(128*(i+1), 128 * (i+2)))


    def forward(self,x):
        for layer in self.encode_net:
            x = F.relu(layer(x))
        n2 = len(self.layers)
        for i in  range(n - 1)]
            x = F.relu(layer[i](x))
        x = F.sigmoid(layer[n-1](x))
        return x


    def loss(self,y,x): #@TODO CHECK HERE
        train_z = self.encode_net(x)
        samples = torch.randn([200,self.num_latent])
        loss_mmd = compute_mmd(samples, train_z)
        loss_nll = torch.mean((y - x)**2)
        return loss_mmd + loss_nll

def compute_kernel(x,y):
    x_size, dim = x.size()
    y_size = y.size()[0]
    tiled_x = x.reshape(x_size,1,dim).repeat([1,y_size,1])
    tiled_y = y.reshape(1,y_size,dim).repeat([x_size,1,1])
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,axis=2) / float(dim))


def compute_mmd(x,y):
    x_kernel = compute_kernel(x,x)
    y_kernel = compute_kernel(y,y)
    xy_kernel = compute_kernel(x,y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

def train_mmd_vae(genotype, params): #@TODO CHECK HERE
    model = MMD_VAE(**params)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for i in range(100): #num epochs?
        optimizer.zero_grad()
        output = model(genotype)
        loss = model.loss(output, genotype)
