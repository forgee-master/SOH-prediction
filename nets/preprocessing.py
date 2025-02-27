import torch
import torch.nn as nn
import torch.nn.functional as F


class PreProcessing(nn.Module):

    def __init__(self, input_type):
        super(PreProcessing, self).__init__()
        self.input_type = input_type

        if self.input_type in  ('charge', 'partial_charge'): # charge_data (N,4,128)
            self.net = nn.Conv1d(in_channels=4,out_channels=4,kernel_size=1)
        elif self.input_type in ("features"):  # features (N,1,67)
            self.net = nn.Linear(67,128*4) #(N,128*4)

        self.layer_norm = nn.BatchNorm1d(4)

    
    def forward(self,x):
        
        x = self.net(x)
        x = x.view(-1,4,128)
        out = self.layer_norm(x)
        return out