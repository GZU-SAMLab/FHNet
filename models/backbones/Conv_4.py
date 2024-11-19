import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers1 = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers2 = nn.Sequential(
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers3 = nn.Sequential(
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers4 = nn.Sequential(
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self,inp):

        out1 = self.layers1(inp)
        out2 = self.layers2(out1)
        out3 = self.layers3(out2)
        out4 = self.layers4(out3)
        return out1, out2, out3, out4

if __name__ == '__main__':
    model = BackBone()
    inp = torch.randn(300,3,84,84)
    out1, out2, out3, out4 = model(inp)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)