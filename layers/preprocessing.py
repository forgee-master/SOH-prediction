import torch
import torch.nn as nn
import torch.nn.functional as F


class PreProcessingNet(nn.Module):

    def __init__(self, args):
        super(PreProcessingNet, self).__init__()
        self.in_features = args.features
        self.out_features = args.post_features
        self.seq_len = args.seq_len

        if args.input_type in  ('charge', 'partial_charge'): # charge_data (N,4,128)
            self.net = nn.Conv1d(in_channels=self.in_features,out_channels=self.out_features,kernel_size=1)
        elif args.input_type in ("handcraft_features"):  # features (N,1,67)
            self.net = nn.Conv1d(in_channels=self.in_features,out_channels=self.out_features,kernel_size=1) #(N,128*4)

        self.batch_norm = nn.LayerNorm(self.seq_len)

    
    def forward(self,x):
        print(x.shape)
        x = self.net(x)
        x = x.view(-1,self.out_features,self.seq_len)
        x = self.batch_norm(x)
        return x