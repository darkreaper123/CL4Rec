import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.nn import init
from torch.nn import Parameter

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

            
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        #norm = x.pow(2).sum(dim=-1).sqrt()
        #x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        #if self.use_task_mask:
        h1 = self.ln1(self.fc1(x))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        mu = self.fc_mu(h5)
        log_var = self.fc_logvar(h5)
        return mu, log_var
    

class VAE(nn.Module):
    def __init__(self, args, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, index = None, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True, index_max = None, user_ratings_out = None):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        if calculate_loss:
            if user_ratings_out is None:
                mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            else:
                mll = (F.log_softmax(x_pred, dim=-1) * user_ratings_out).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar)).sum(dim=-1).mean()
            negative_elbo = -(mll - 0)
            
            return (mll, 0), negative_elbo
            
        else:
            return x_pred