import torch
import torch.nn as nn
import torch.nn.functional as F

# input:   3 x 32 x 32
# output:  3 x 128 x 128

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        _skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x + _skip

class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pix_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pix_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.block4 = ResidualBlock()
        self.block5 = ResidualBlock()

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample1 = UpsampleBlock()
        self.upsample2 = UpsampleBlock()

        self.last = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.first(x)
        _skip = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.mid(x)
        x = x + _skip

        x = self.upsample1(x)
        x = self.upsample2(x)

        x = self.last(x)

        return x

class DBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(DBlock, self).__init__()

        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(outchannel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, xh=128, xw=128):
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.block2 = DBlock(64, 64, 2)
        self.block3 = DBlock(64, 128, 1)
        self.block4 = DBlock(128, 128, 2)
        self.block5 = DBlock(128, 256, 1)
        self.block6 = DBlock(256, 256, 2)
        self.block7 = DBlock(256, 512, 1)
        self.block8 = DBlock(512, 512, 2)

        self.block9 = nn.Sequential(
            nn.Linear(512 * (xh//16) * (xw//16), 1204),
            nn.LeakyReLU(0.2),
            nn.Linear(1204, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        return x


