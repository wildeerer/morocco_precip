import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=.5)
        self.prelu = nn.ReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=.5)
        
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv2.bias)


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class Generator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.initial_prelu = nn.PReLU(64)
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
        
        self.post_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.post_bn = nn.BatchNorm2d(64, momentum=.5)
        
        self.up1_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_prelu = nn.PReLU(128)
        
        self.up2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=3, mode='nearest')
        self.up2_prelu = nn.PReLU(128)
        
        self.up3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3_prelu = nn.PReLU(128)
        
        self.final_conv = nn.Conv2d(128, 1, kernel_size=9, padding=4)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_prelu(x)
        gen_model = x
        
        x = self.res_blocks(x)
        
        x = self.post_conv(x)
        x = self.post_bn(x)
        x = gen_model + x
        
        x = self.up1_conv(x)
        x = self.up1(x)
        x = self.up1_prelu(x)
        
        x = self.up2_conv(x)
        x = self.up2(x)
        x = self.up2_prelu(x)
        
        x = self.up3_conv(x)
        x = self.up3(x)
        x = self.up3_prelu(x)
        
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    gen = Generator(in_channels=1)
    
    test_input = torch.randn(1, 1, 13, 16)
    output = gen(test_input)
    print(test_input)
    print(output)