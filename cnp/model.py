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

    def forward(self, x_obs, y_obs, x_tst):
        '''
        x_obs: [B, O, x_dim]
        y_obs: [B, O, y_dim]
        x_tst: [B, T, x_dim]
        ---
        out: [B, T, out_dim] # distr. params for p(y_tst | (x_obs, y_obs), x_tst)
        '''
        r_all = self.encoder(x_obs, y_obs)
        r = torch.mean(r_all, dim=1) # aggregation
        out = self.decoder(x_tst, r)
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
        x: [B, O, x_dim]
        y: [B, O, y_dim]
        ---
        r_all: [B, O, r_dim]
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
        x: [B, T, x_dim]
        r: [B, r_dim]
        ---
        out: [B, T, out_dim]
        '''
        r_expand = r.unsqueeze(dim=1).expand(-1, x.size(1), -1) # [B, T, r_dim]
        xr = torch.cat(x, r_expand, dim=2)
        out = self.layers(xr)
        return out
