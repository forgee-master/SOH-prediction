import torch
import torch.nn as nn


class Model(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len # L
        self.in_features = args.post_features # c
        self.net_hidden_size = args.mlp_hidden_size
        self.pred_hidden_size = args.pred_hidden_size

        self.net = nn.Sequential(
            nn.Linear(self.seq_len * self.in_features, self.net_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.net_hidden_size, self.seq_len),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_hidden_size),
            nn.ReLU(),
            nn.Linear(self.pred_hidden_size, 1)
        )

    # x: (N, c, L)
    def forward(self, x):
        x = x.view(-1,self.in_features*self.seq_len)
        fea = self.net(x)
        out = self.predictor(fea)
        return out