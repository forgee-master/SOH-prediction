import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DiscreteCosine import DiscreteCosineTransform


class LowRank(nn.Module):
    """
    Implements a low-rank approximation layer using two smaller weight matrices (A and B).
    This reduces the number of parameters compared to a full-rank layer.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        # Initialize weight matrices A (in_features x rank) and B (rank x out_features)
        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)
        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))

        # Initialize bias if required
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        # Apply low-rank transformation: X * A * B
        out = x @ self.A
        out = out @ self.B
        if self.bias:
            out += self.b  # Add bias if enabled
        return out
    


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        
        self.ffn1 = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,1),
            #nn.ReLU()
            )
        
        self.ffn2 = nn.Sequential(
            LowRank(128,256,rank=30),
            nn.ReLU(),
            LowRank(256,1, rank=16)
        )


    def forward(self, x):
         x = DiscreteCosineTransform.apply(x) /x.shape[-1]
         x = x.permute(0,2,1)
         x = self.ffn1(x)
         x = x.permute(0,2,1)
         x = self.ffn2(x)
         x.squeeze_(-1)
         return x




