import torch
import torch.nn as nn
import torch.nn.functional as F

def NLLloss(y, mean, var):
    return torch.mean(torch.log(var)+ (y - mean)**2/(2*var))


class CNP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, out_dim=2, r_dim=128):
        super(CNP, self).__init__()
        self.encoder = Encoder(x_dim, y_dim, r_dim)
        self.decoder = Decoder(x_dim, r_dim, out_dim)

    def forward(self, x_obs, y_obs, x_tar):
        '''
        x_obs: [B, N, x_dim]
        y_obs: [B, N, y_dim]
        x_tar: [B, M, x_dim]
        ---
        out: [B, M, out_dim] # distr. params for p(y_tar | (x_obs, y_obs), x_tar)
        '''
        r_all = self.encoder(x_obs, y_obs)
        r = torch.mean(r_all, dim=1) # aggregation
        out = self.decoder(x_tar, r)
        return out


class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim):
        super(Encoder, self).__init__()
        h_dim = 100
        self.layers = nn.Sequential(
            nn.Linear(x_dim+y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, r_dim)
        )

    def forward(self, x, y):
        '''
        x: [B, N, x_dim]
        y: [B, N, y_dim]
        ---
        r_all: [B, N, r_dim]
        '''
        xy = torch.cat([x, y], dim=2)
        r_all = self.layers(xy)
        return r_all


class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, out_dim):
        super(Decoder, self).__init__()
        h_dim = 100
        self.layers = nn.Sequential(
            nn.Linear(x_dim+r_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x, r):
        '''
        x: [B, M, x_dim]
        r: [B, r_dim]
        ---
        out: [B, M, out_dim]
        '''
        r_expand = r.unsqueeze(dim=1).expand(-1, x.size(1), -1) # [B, M, r_dim]
        xr = torch.cat([x, r_expand], dim=2)
        out = self.layers(xr)
        return out
