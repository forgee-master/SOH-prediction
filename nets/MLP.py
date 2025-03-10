import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding


class Model(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len # L
        self.enc_in = args.features # c
        
        self.feature_converter = nn.Sequential(
            nn.Conv1d(
                in_channels=self.enc_in, 
                out_channels=self.enc_in*2, 
                kernel_size=1, 
                groups=self.enc_in),
        )


        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.seq_len*self.enc_in*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )


    # x: (N, c, L)
    def forward(self, x):
        seq_mean = x.mean(dim=-1, keepdim=True) # (N, 1, L)
        x = x - seq_mean # (N, c, L)
        x = self.feature_converter(x)
        out = self.net(x)
        return out