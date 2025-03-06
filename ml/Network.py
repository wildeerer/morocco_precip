import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Conv2d(in_features, 64, kernel_size=3)
        self.block2 = nn.BatchNorm2d(64)
        self.block3 = nn.ReLU()
        self.block4 = nn.Conv2d(64, out_features, kernel_size=3)
        self.block5 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return 



class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsamplingBlock, self).__init__()
        self.block1 = nn.Conv2d()
        self.block2 = nn.UpsamplingBilinear2d()
        self.block3 = nn.ReLU()
