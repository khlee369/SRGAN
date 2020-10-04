import torch
import torch.nn as nn
import torch.nn.functional as F

# input:   3 x 16 x 16
# output:  3 x 128 x 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)

        self.B_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.B_bn = nn.BatchNorm2d(64)

        self.P_conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pix_shuffle = nn.PixelShuffle(2)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.PReLU()(x)
        fx = x
        tx = x

        # B residual blocks
        for _ in range(5):
            x = self.B_conv(tx)
            x = self.B_bn(x)
            x = nn.PReLU()(x)
            x = self.B_conv(x)
            x = self.B_bn(x)
            x = tx + x
            tx = x

        x = self.B_conv(tx)
        x = self.B_bn(x)
        x = fx + x

        # Pixel Shuffler Upsampling
        # There are 2 Upsampling in Paper
        # My model Upsample 3 times
        for _ in range(3):
            x = self.P_conv(x)
            x = self.pix_shuffle(x)
            x = nn.PReLU()(x)

        x = self.conv2(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        pass
