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

class MaskLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, n_tasks = 3, limits = None):
        super().__init__(in_features = in_features, out_features = out_features, bias = bias,
                 device=None, dtype=None)
        self.n_tasks = n_tasks
        self.limits = limits
        self.check_bias = bias
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mask_weights = nn.ParameterList([nn.Parameter(torch.empty((out_features,1), **factory_kwargs)) for i in range(n_tasks)])
        for i in range(n_tasks):
            self.register_parameter('mask_weight_{}'.format(i), self.mask_weights[i])

        self.reset_mask_parameters()  
    def reset_mask_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for index in range(self.n_tasks):
            init.kaiming_uniform_(self.mask_weights[index], a=math.sqrt(5))
    def forward(self, input, index = None, use_binary_mask = None, index_max = None):
        if index is not None:
            #if index == 1:
                #print(torch.sigmoid(self.mask_weights[1]))
                #print(torch.sigmoid(self.mask_weights[0]))
            if True:
                mask_weight = torch.sigmoid(self.mask_weights[index])
                if self.limits is not None and use_binary_mask is not None:
                    if use_binary_mask[index]:
                        mask_weight = nn.Threshold(self.limits[index], 0.)(mask_weight)
                        mask_weight = nn.Threshold(-self.limits[index], 1.)(-mask_weight)
                new_weight = self.weight * mask_weight
                if self.check_bias is not None:
                    if index == 0:
                        bias = self.bias
                    else:
                        with torch.no_grad():
                            bias = self.bias
                else:
                    bias = None
                output = F.linear(input, new_weight, bias)
                return output
        if index_max is not None:
            if True:
                mask_weights = []
                for i in range(index_max + 1):
                    mask_weight = torch.sigmoid(self.mask_weights[i])
                    if use_binary_mask[i]:
                        mask_weight = nn.Threshold(self.limits[i], 0.)(mask_weight)
                        mask_weight = nn.Threshold(-self.limits[i], 1.)(-mask_weight)
                    mask_weights.append(mask_weight)
                #print(mask_weights)
                mask_weights, _ = torch.max(torch.concat(mask_weights, dim = -1), dim = -1)
                if self.check_bias is not None:
                    bias = self.bias
                else:
                    bias = None
                
                return F.linear(input, self.weight * mask_weights.view(-1, 1), bias)
            else:
                return F.linear(input, self.weight, self.bias)
        
class ReverseMaskLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, n_tasks = 2, limits = None):
        super().__init__(in_features = in_features, out_features = out_features, bias = bias,
                 device=None, dtype=None)
        self.n_tasks = n_tasks
        self.limits = limits
        self.check_bias = bias
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mask_weights = nn.ParameterList([nn.Parameter(torch.empty((1,in_features), **factory_kwargs)) for i in range(n_tasks)])
        for i in range(n_tasks):
            self.register_parameter('reverse_mask_weight_{}'.format(i), self.mask_weights[i])

        self.reset_mask_parameters()  
    def reset_mask_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for index in range(self.n_tasks):
            init.kaiming_uniform_(self.mask_weights[index], a=math.sqrt(5))
    def forward(self, input, index = None, use_binary_mask = None, index_max = None):
        if True:
            if index is not None:
                #if index == 1:
                    #print(torch.sigmoid(self.mask_weights[1]))
                    #print(torch.sigmoid(self.mask_weights[0]))
                if True:
                    mask_weight = torch.sigmoid(self.mask_weights[index])
                    if self.limits is not None and use_binary_mask is not None:
                        if use_binary_mask[index]:
                            mask_weight = nn.Threshold(self.limits[index], 0.)(mask_weight)
                            mask_weight = nn.Threshold(-self.limits[index], 1.)(-mask_weight)
                    new_weight = self.weight * mask_weight
                    if self.check_bias is not None:
                        if index == 0:
                            bias = self.bias
                        else:
                            with torch.no_grad():
                                bias = self.bias
                    else:
                        bias = None
                    output = F.linear(input, new_weight, bias)
                    return output
            if index_max is not None:
                if True:
                    mask_weights = []
                    for i in range(index_max + 1):
                        mask_weight = torch.sigmoid(self.mask_weights[i])
                        if use_binary_mask[i]:
                            mask_weight = nn.Threshold(self.limits[i], 0.)(mask_weight)
                            mask_weight = nn.Threshold(-self.limits[i], 1.)(-mask_weight)
                        mask_weights.append(mask_weight)
                    mask_weights, _ = torch.max(torch.concat(mask_weights, dim = 0), dim = 0)
                    if self.check_bias is not None:
                        bias = self.bias
                    else:
                        bias = None

                    return F.linear(input, self.weight * mask_weights.view(1,-1), bias)
                else:
                    return F.linear(input, self.weight, self.bias)
            
            
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1, use_task_mask = False, limits = None, vae_using_ufo_space = True):
        super(Encoder, self).__init__()
        self.use_task_mask = use_task_mask
        self.vae_using_ufo_space = vae_using_ufo_space
        if self.use_task_mask and self.vae_using_ufo_space:
        #if False:
            self.fc1 = MaskLinear(input_dim, hidden_dim, n_tasks = len(limits), limits = limits)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.limits = limits
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
        
    def forward(self, x, dropout_rate, index, use_binary_mask, index_max):
        #norm = x.pow(2).sum(dim=-1).sqrt()
        #x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        #if self.use_task_mask:
        try:
            h1 = self.ln1(swish(self.fc1(x, index, use_binary_mask, index_max)))
        except:
            h1 = self.ln1(self.fc1(x))
        if index == 0:
            h2 = self.ln2(swish(self.fc2(h1) + h1))
            h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
            h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
            h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
            mu = self.fc_mu(h5)
            log_var = self.fc_logvar(h5)
        else:
            with torch.no_grad():
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
        
        self.use_task_mask = args.use_task_mask
        self.use_binary_mask = [False, False, False]
        self.limits = [float(elem) for elem in args.limits.split(',')]

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim, use_task_mask = self.use_task_mask, limits = self.limits, vae_using_ufo_space = args.vae_using_ufo_space)
        #self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        if self.use_task_mask and args.vae_using_ufo_space:
        #if False:
            self.decoder = ReverseMaskLinear(latent_dim, input_dim, n_tasks = len(self.limits), limits = self.limits)
        else:
            self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, index = None, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True, index_max = None, user_ratings_out = None):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate, index = index, use_binary_mask = self.use_binary_mask, index_max = index_max)    
        z = self.reparameterize(mu, logvar)
        try:
            x_pred = self.decoder(z, index, self.use_binary_mask, index_max)
        except:
            if False:
                check = index is not None
                if check:
                    check = index > 0
                if check:
                    with torch.no_grad():
                        x_pred = self.decoder(z)
                else:
                    x_pred = self.decoder(z)
            else:
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