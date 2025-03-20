import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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
        self.A = nn.Parameter(nn.init.orthogonal_(wA))
        self.B = nn.Parameter(nn.init.orthogonal_(wB))

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
        self.seq_len = args.seq_len # L
        self.enc_in = args.features # c
        self.enable_Haar = True
        

        # self.feature_converter = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.enc_in, 
        #         out_channels=self.enc_in*2, 
        #         kernel_size=1, 
        #         groups=self.enc_in),
        # )

        # Define low-pass filter for Haar decomposition (averaging adjacent values)
        self.low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)
        self.low_pass_filter = self.low_pass_filter.reshape(1, 1, -1).repeat(self.enc_in, 1, 1)

        self.high_pass_filter = torch.tensor([-1, 1], dtype=torch.float32) / math.sqrt(2)
        self.high_pass_filter = self.high_pass_filter.reshape(1, 1, -1).repeat(self.enc_in, 1, 1)

        # Adjust input length if Haar decomposition is enabled
        if self.enable_Haar:
            in_len = (self.seq_len // 2) + 1 if (self.seq_len % 2) != 0 else (self.seq_len // 2)
        else:
            in_len = self.seq_len  # No Haar decomposition, use full sequence length

        self.net = nn.Sequential(
            LowRank(in_len * 6, 128, rank=30),
            nn.ReLU(),
            LowRank(128, 1, rank=15),
        )

    # x: (N, c, L)
    def forward(self, x):
        
        # Apply Haar transformation (low-pass filtering) if enabled
        if self.enable_Haar:
            if self.seq_len % 2 != 0:
                x = F.pad(x, (0, 1))  # Pad if sequence length is odd
            
            self.low_pass_filter = self.low_pass_filter.to(x.device)  # Move filter to same device as input
            self.high_pass_filter = self.high_pass_filter.to(x.device)

            x1 = F.conv1d(input=x, weight=self.low_pass_filter, stride=2, groups=self.enc_in)  # Apply low-pass filter
            x2 = F.conv1d(input=x, weight=self.high_pass_filter, stride=2, groups=self.enc_in)
            x = torch.cat((x1, x2), dim=1)
        

        x =  torch.flatten(x, start_dim=1, end_dim=-1) # Flatten output for fully connected layers
        out = self.net(x)
         
        return out




