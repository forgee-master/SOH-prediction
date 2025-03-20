import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        padding = 1  # Manually set to mimic 'same' padding
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel or stride > 1:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        return self.relu(out)



class Model(nn.Module):
    '''
    input shape: (N,4,128)
    '''
    def __init__(self, args):
        super(Model,self).__init__()

        self.seq_len = args.seq_len # L
        self.enc_in = args.features # c
        
        self.feature_converter = nn.Sequential(
            nn.Conv1d(
                in_channels=self.enc_in, 
                out_channels=self.enc_in*2, 
                kernel_size=1, 
                groups=self.enc_in),
        )

        self.resnet = nn.Sequential(
            ResBlock(input_channel=self.enc_in*2, output_channel=16, stride=1),
            ResBlock(input_channel=16, output_channel=32, stride=2),
            ResBlock(input_channel=32, output_channel=64, stride=2),
            ResBlock(input_channel=64, output_channel=96, stride=2),
            ResBlock(input_channel=96, output_channel=128, stride=2),
            ResBlock(input_channel=128, output_channel=256, stride=2),
        )

        flattened_dim = 256 * (self.seq_len // 32)  # Account for downsampling

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        seq_mean = x.mean(dim=-1, keepdim=True)  # Normalize over time
        x = x - seq_mean
        x = self.feature_converter(x)
        x = self.resnet(x)
        out = self.net(x)
        return out