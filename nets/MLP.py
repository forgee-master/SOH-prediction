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
        
        self.feature_converter = nn.Linear(self.enc_in, self.enc_in*2)

        self.pos_embed = PositionalEmbedding(self.enc_in*2)

        self.net = nn.Sequential(
            nn.Linear(self.seq_len*self.enc_in*2, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )


    # x: (N, c, L)
    def forward(self, x):
        seq_mean = x.mean(dim=1, keepdim=True) # (N, 1, L)
        x = x - seq_mean # (N, c, L)
        x = self.feature_converter(x.permute(0,2,1))
        #x = x + self.pos_embed(x)
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = torch.unsqueeze(out, -1)
        return out
    

## Test Loss MAE:4.497 MSE:0.035 MAPE:4.876 with Standard Normalization