import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.autograd import Variable
from torch import nn
import torch
from .vanilla_vae import VanillaVAE
from .cvae import ConditionalVAE

class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc4 = torch.nn.Linear(16, 8)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.scale = 4.0

        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 64)
        self.fc3 = torch.nn.Linear(64, 512)
        self.fc4 = torch.nn.Linear(512, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.scale*torch.tanh(self.fc4(x)) # -scale ~ scale
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, n_channels=6, len_sw=300):
        super().__init__()        
        self.enc = Encoder(n_channels*len_sw)
        self.dec = Decoder(n_channels*len_sw)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, **kwargs):
        x_in = x.detach().clone()
        len_sw = x.shape[1]
        n_channels = x.shape[2]
        x = x.reshape(-1,len_sw*n_channels)
        x = self.enc(x)
        x = self.dec(x)
        x = x.reshape(-1,len_sw,n_channels)
        return self.criterion(x,x_in), x


hidden_dims_vae = [16,32,64,128,256]

class VAE(torch.nn.Module):
    def __init__(self, n_channels=6):
        super().__init__()
        self.vae = VanillaVAE(n_channels,latent_dim=128, hidden_dims = hidden_dims_vae.copy())
    
    def forward(self, x, **kwargs):
        x = nn.functional.pad(torch.permute(x,(0,2,1)),(0,20), "constant", 0) #(N,C,W)
        output = self.vae(x, **kwargs)
        x = torch.permute(output[0],(0,2,1))
        config = {'M_N': 0.00025}
        return self.vae.loss_function(*output,**config)['loss'], x[:,0:300,:]
    
    def sample(self, nsamples, **kwargs):
        x = self.vae.sample(nsamples,kwargs['device'])
        x = torch.permute(x,(0,2,1))
        return x[:,0:300,:]
        
    
class CVAE(torch.nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.cvae = ConditionalVAE(n_channels,latent_dim=128, hidden_dims = hidden_dims_vae.copy(),num_classes=10)
    
    def forward(self, x, **kwargs):
        x = nn.functional.pad(torch.permute(x,(0,2,1)),(0,20), "constant", 0)
        output = self.cvae(x, **kwargs)
        x = torch.permute(output[0],(0,2,1))
        config = {'M_N': 0.00025}
        return self.cvae.loss_function(*output,**config)['loss'], x[:,0:300,:]

    def sample(self, nsamples, **kwargs):
        x = self.cvae.sample(nsamples,kwargs['device'],**kwargs)
        x = torch.permute(x,(0,2,1))
        return x[:,0:300,:]

