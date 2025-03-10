import torch
import torch.nn as nn


class Model(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self, args):
        super(Model, self).__init__()

        self.seq_len = args.seq_len # L
        self.enc_in = args.features # c
        hidden_size = args.hidden_size
        bidirectional = args.bidirectional
        num_layers = args.num_layers

        self.feature_converter = nn.Sequential(
            nn.Conv1d(
            in_channels=self.enc_in, 
            out_channels=self.enc_in*2, 
            kernel_size=1, 
            groups=self.enc_in),
        )

        self.states = nn.LSTM(input_size=self.enc_in*2 ,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0.2)
        
        if bidirectional:
            hidden_size *= 2

        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # x: (N, c, L)
    def forward(self, x):

        seq_mean = x.mean(dim=1, keepdim=True) # (N, 1, L)
        x = x - seq_mean # (N, c, L)
        x = self.feature_converter(x)

        x = x.permute(0,2,1)
        x = self.states(x)[0]
        x = x[:,-1,:]
        out = self.net(x)

        return out