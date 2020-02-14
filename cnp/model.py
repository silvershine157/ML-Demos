import torch
import torch.nn as nn
import torch.nn.functional as F

def NLLloss(y, mean, var):
    return torch.mean(torch.log(var)+ (y - mean)**2/(2*var))

# TODO -- Implement CNP model for 1d regression

class CNP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128):
        super(CNP, self).__init__()
        self.encoder = Encoder(x_dim, y_dim, r_dim)
        self.decoder = Encoder(x_dim, y_dim, r_dim)

    # Output : mean, variance
    def forward(self, x_ctx, y_ctx, x_obs):
        '''
        x_ctx: [B, C, x_dim]
        y_ctx: [B, C, y_dim]
        x_obs: [B, O, x_dim]
        ---
        mean: [B, O, y_dim]
        var: [B, O, y_dim]
        '''
        r_all = self.encoder(x_ctx, y_ctx)
        r = torch.mean(r_all, dim=1) # aggregation
        mean, var = self.decoder(x_obs, r)
        return mean, var

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
        x: [B, C, x_dim]
        y: [B, C, y_dim]
        ---
        r_all: [B, C, r_dim]
        '''
        xy = torch.cat([x, y], dim=2)
        r_all = self.layers(xy)
        return r_all

class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim):
        super(Decoder, self).__init__()
        self.y_dim = y_dim
        h_dim = 100
        self.layers = nn.Sequential(
            nn.Linear(x_dim+r_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2*y_dim)
        )

    def forward(self, x, r):
        '''
        x: [B, O, x_dim]
        r: [B, r_dim]
        ---
        mean: [B, O, y_dim]
        var: [B, O, y_dim]
        '''
        r_expand = r.unsqueeze(dim=1).expand(-1, x.size(1), -1) # [B, O, r_dim]
        xr = torch.cat(x, r_expand, dim=2)
        out = self.layers(xr)
        mean = out[:, :, :self.y_dim]
        var = torch.exp(out[:, :, self.y_dim:])
        return mean, var