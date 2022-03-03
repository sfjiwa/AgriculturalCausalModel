import torch;torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; 
from scipy.special import gamma, factorial
from torch.nn.functional import normalize
device ='cpu'

class Encoder(nn.Module):#variational encoder
    def __init__(self, latent_dims):
        super().__init__()
        self.circ = torch.nn.Parameter(torch.ones(3500))
        self.a = torch.nn.Parameter(torch.zeros(3500))
        self.linear1 = nn.Linear(195*179*3, 3500)
        self.linear3 = nn.Linear(195*179*3,3500)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.N1 = torch.distributions.Normal(0,1)
        self.N1.loc = self.N1.loc
        self.N1.scale = self.N1.scale
        self.kl = 0
    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        sigma = torch.exp(self.linear3(xy))
        xy = self.linear1(xy)

        mu = self.circ*xy + self.a
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        z = torch.mean(z)
        return z



class Decoder(nn.Module): # a layer in the decoder could represent the oceananic fluxes (this could also be in the bcgproxy)
    def __init__(self, latent_dims):
        super().__init__()
        self.par1 = torch.nn.Parameter(torch.ones(195*179*3))
        self.par2 = torch.nn.Parameter(torch.zeros(195*179*3))
    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.par2 + torch.mean(zy)*self.par1


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    def forward(self, x):
        z = self.encoder(torch.sigmoid(x))
        return self.decoder(z)
    def latentForward(self,x):
        return self.encoder(x)
    def deco(self,x):
        return self.decoder(x)


def train(autoencoder,data, indexData,epochs=1):
    lam = 1 #parameter lambda
    opt1 = torch.optim.Adam(autoencoder.parameters(),lr=0.0001)
    vae_loss = []
    lin_loss = []
    estimatedindex = []
    actualindex = []
    print(epochs)
    for epoch in range(epochs):
        
        # we train in alternate steps, first fixing h 
        for x,y1 in zip(data,indexData): #range data to access both this should def. be batched n done using sgd
            
            x = x.float().to(device) # GPU
            y1 = y1.float().to(device)
            
            opt1.zero_grad()
            x_hat = autoencoder.forward(x)
            z = autoencoder.latentForward(x)
            estimatedindex.append(torch.mean(z).detach().numpy())
            actualindex.append(torch.mean(y1).detach().numpy())
            loss1 = 0.5*((x - x_hat)**2).mean() + 2*autoencoder.encoder.kl +2*(y1-z)**2

            loss1.backward()
            opt1.step()
        print('iteration:')
        print(epoch)
        print(loss1)
        print(np.mean(estimatedindex))
        plt.scatter(actualindex,estimatedindex)
        plt.show()
    return autoencoder




latent_dims=1
autoencoder = Autoencoder(latent_dims)
autoencoder.load_state_dict(torch.load('indexVAE.pth'))
autoencoder.to(device)
autoencoder.eval()
data = torch.utils.data.DataLoader(torch.nan_to_num(torch.load('index1.pt')))
indexData = torch.utils.data.DataLoader(torch.nan_to_num(torch.load('usage.pt')))
estimatedindex = []
actualindex = []
c1 = 0
t1 = 0
jjs = []
aces = []
for jj in range(70,90,2):
    for x,y1 in zip(data,indexData): #range data to access both this should def. be batched n done using sgd          
        y1 = y1.float().to(device)
        x = x.float().to(device) # GPU
        x_hat = autoencoder.forward(x)
        z = autoencoder.latentForward(x)
        t1+=1
        if (z>jj*0.01):
            c1+=1
            actualindex.append(y1.detach().numpy())
            estimatedindex.append(z.detach().numpy())
    print(np.shape(actualindex))
    print(np.shape(estimatedindex))
    plt.scatter(actualindex,estimatedindex)
    plt.show()
    print(np.mean(actualindex))
    print("p")

    data1 = torch.utils.data.DataLoader(torch.nan_to_num(torch.load('index1f.pt')))
    es1 = []
    c2 = 0
    t2 = 0
    for x in data1: #test on future data         
        x = x.float().to(device) # GPU
        x_hat = autoencoder.forward(x)
        z = autoencoder.latentForward(x)
        t2+=1
        if(z>jj*0.01):
            c2+=1

    es1.append(z.detach().numpy())
    print(np.mean(es1))
    print("y given hist ")
    print(c1/t1)
    print("y given rcp")
    print(c2/t2)
    print("ACE")
    print(c2/t2-c1/t1)
    print(jj)
    jjs.append(jj)
    aces.append(c2/t2-c1/t1)

plt.scatter(jjs,aces)
plt.show()

