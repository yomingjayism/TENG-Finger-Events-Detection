'''
Created on Thu Dec 12 14:58:47 2024

@author: Kendrick
'''

import torch
import torch.nn as nn

class ConvNormReluLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvNormReluLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, signals):
        signals = self.conv1d(signals)
        signals = self.norm(signals)
        return self.act(signals)

class GestureWaveConv(nn.Module):
    def __init__(
        self, 
        in_channels=None, 
        dims=None, 
        out_channels=None, 
        config=None
        ) -> None:
        super().__init__()
        
        if config is not None:
            self.config = config
            self.in_channels = config.in_channels
            self.dims = config.dims
            self.out_channels = config.out_channels
        
        else:
            self.in_channel = in_channels
            self.dims = dims
            self.out_channels = out_channels
            
        self.conv1 = ConvNormReluLayer(in_channels=self.in_channels, out_channels=self.dims[0], kernel_size=(7,), padding="same")
        self.conv2 = ConvNormReluLayer(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=(13,), padding="same")
        self.conv3 = ConvNormReluLayer(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=(19,), padding="same")
        self.conv4 = ConvNormReluLayer(in_channels=self.dims[2], out_channels=self.out_channels, kernel_size=(19,), padding="same")
        
    def forward(self, signal:torch.tensor, labels:torch.tensor=None) -> dict:
        x = self.conv1(signal)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        pred = nn.ReLU()(x)

        if labels is not None:
            loss = nn.MSELoss()(pred, labels)
        else:
            loss = None
            
        outputs = {"pred":pred, "loss":loss}
        return outputs