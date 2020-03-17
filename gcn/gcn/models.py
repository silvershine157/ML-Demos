import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # TODO 
        self.w1 = nn.Linear(nfeat, nhid)
        self.w2 = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        # TODO
        '''
        x: [N, nfeat]
        adj: [N, N]
        ---
        output: [N, nclass]
        '''
        h1 = F.relu(self.w1(torch.mm(adj, x))) # [N, nhid]
        output = F.softmax(self.w2(torch.mm(adj, h1)), dim=1)
        return output
