import torch
import torch.nn as nn
import torch.nn.functional as F

# input:   3 x 32 x 32
# output:  3 x 256 x 256

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)

        self.rblocks = 5
        self.B_convs = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) for i in range(2*self.rblocks)]
        self.B_bns = [nn.BatchNorm2d(64) for i in range(2*self.rblocks)]

        self.pblocks = 2
        self.P_convs = [nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1) for i in range(self.pblocks)]
        self.pix_shuffle = nn.PixelShuffle(2)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.PReLU().cuda()(x)
        fx = x
        tx = x

        # B residual blocks
        for i in range(0, 2*self.rblocks, 2):
            x = self.B_convs[i](tx)
            x = self.B_bns[i](x)
            x = nn.PReLU().cuda()(x)
            x = self.B_convs[i+1](x)
            x = self.B_bns[i+1](x)
            x = tx + x
            tx = x

        x = self.B_conv(tx)
        x = self.B_bn(x)
        x = fx + x

        # Pixel Shuffler Upsampling
        # There are 2 Upsampling in Paper
        # My model Upsample 3 times
        for i in range(2):
            x = self.P_convs[i](x)
            x = self.pix_shuffle(x)
            x = nn.PReLU().cuda()(x)

        x = self.conv2(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, xh = 128, xw = 128):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.LReLU = nn.LeakyReLU(0.2)
        self.bn64 = nn.BatchNorm2d(64)

        self.conv128s1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv128s2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)

        self.conv256s1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv256s2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn256_1 = nn.BatchNorm2d(256)
        self.bn256_2 = nn.BatchNorm2d(256)

        self.conv512s1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv512s2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn512_1 = nn.BatchNorm2d(512)
        self.bn512_2 = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512 * (xh//16) * (xw//16), 1204)
        self.lin2 = nn.Linear(1204, 1)

        self.blocks = [[self.conv128s1, self.conv128s2, self.bn128_1, self.bn128_2], 
                       [self.conv256s1, self.conv256s2, self.bn256_1, self.bn256_2], 
                       [self.conv512s1, self.conv512s2, self.bn512_1, self.bn512_2]]

    def forward(self, x):
        x = self.conv1(x)
        x = self.LReLU(x)
        x = self.conv2(x)
        x = self.bn64(x)
        x = self.LReLU(x)

        for s1, s2, bn1, bn2 in self.blocks:
            x = s1(x)
            x = bn1(x)
            x = self.LReLU(x)

            x = s2(x)
            x = bn2(x)
            x = self.LReLU(x)

        x = self.lin1(x.view(x.size()[0], -1))
        x = self.LReLU(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)

        return x
