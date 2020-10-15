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

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.conv964 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
#         self.PReLU964 = nn.PReLU().cuda()

#         self.B_conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn1_1 = nn.BatchNorm2d(64)
#         self.B_PReLU1 = nn.PReLU().cuda()
#         self.B_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn1_2 = nn.BatchNorm2d(64)

#         self.B_conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn2_1 = nn.BatchNorm2d(64)
#         self.B_PReLU2 = nn.PReLU().cuda()
#         self.B_conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn2_2 = nn.BatchNorm2d(64)

#         self.B_conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn3_1 = nn.BatchNorm2d(64)
#         self.B_PReLU3 = nn.PReLU().cuda()
#         self.B_conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn3_2 = nn.BatchNorm2d(64)

#         self.B_conv4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn4_1 = nn.BatchNorm2d(64)
#         self.B_PReLU4 = nn.PReLU().cuda()
#         self.B_conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn4_2 = nn.BatchNorm2d(64)

#         self.B_conv5_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn5_1 = nn.BatchNorm2d(64)
#         self.B_PReLU5 = nn.PReLU().cuda()
#         self.B_conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.B_bn5_2 = nn.BatchNorm2d(64)

#         self.conv364 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn364 = nn.BatchNorm2d(64)

#         self.P_conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
#         self.P_PReLU1 = nn.PReLU().cuda()
#         self.P_conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
#         self.P_PReLU2 = nn.PReLU().cuda()

#         self.pix_shuffle = nn.PixelShuffle(2)

#         self.conv93 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

#     def forward(self, x):
#         x = self.conv964(x)
#         x = self.PReLU964(x)
#         fx = x
#         tx = x

#         # B residual blocks
#         x = self.B_conv1_1(x)
#         x = self.B_bn1_1(x)
#         x = self.B_PReLU1(x)
#         x = self.B_conv1_2(x)
#         x = self.B_bn1_2(x)
#         x = tx + x
#         tx = x

#         x = self.B_conv2_1(x)
#         x = self.B_bn2_1(x)
#         x = self.P_PReLU2(x)
#         x = self.B_conv2_2(x)
#         x = self.B_bn2_2(x)
#         x = tx + x
#         tx = x

#         x = self.B_conv3_1(x)
#         x = self.B_bn3_1(x)
#         x = self.B_PReLU3(x)
#         x = self.B_conv3_2(x)
#         x = self.B_bn3_2(x)
#         x = tx + x
#         tx = x

#         x = self.B_conv4_1(x)
#         x = self.B_bn4_1(x)
#         x = self.B_PReLU4(x)
#         x = self.B_conv4_2(x)
#         x = self.B_bn4_2(x)
#         x = tx + x
#         tx = x

#         x = self.B_conv5_1(x)
#         x = self.B_bn5_1(x)
#         x = self.B_PReLU5(x)
#         x = self.B_conv5_2(x)
#         x = self.B_bn5_2(x)
#         x = tx + x
        
#         x = self.conv364(x)
#         x = self.bn364(x)
#         x = x + fx

#         # Upsampling
#         x = self.P_conv1(x)
#         x = self.pix_shuffle(x)
#         x = self.P_PReLU1(x)

#         x = self.P_conv2(x)
#         x = self.pix_shuffle(x)
#         x = self.P_PReLU2(x)

#         x = self.conv93(x)

#         return x

# class Discriminator(nn.Module):
#     def __init__(self, xh = 128, xw = 128):
#         super(Discriminator, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
#         self.LReLU = nn.LeakyReLU(0.2)
#         self.bn64 = nn.BatchNorm2d(64)

#         self.conv128s1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv128s2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
#         self.bn128_1 = nn.BatchNorm2d(128)
#         self.bn128_2 = nn.BatchNorm2d(128)

#         self.conv256s1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv256s2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
#         self.bn256_1 = nn.BatchNorm2d(256)
#         self.bn256_2 = nn.BatchNorm2d(256)

#         self.conv512s1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv512s2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
#         self.bn512_1 = nn.BatchNorm2d(512)
#         self.bn512_2 = nn.BatchNorm2d(512)

#         self.lin1 = nn.Linear(512 * (xh//16) * (xw//16), 1204)
#         self.lin2 = nn.Linear(1204, 1)

#         self.blocks = [[self.conv128s1, self.conv128s2, self.bn128_1, self.bn128_2], 
#                        [self.conv256s1, self.conv256s2, self.bn256_1, self.bn256_2], 
#                        [self.conv512s1, self.conv512s2, self.bn512_1, self.bn512_2]]

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.LReLU(x)
#         x = self.conv2(x)
#         x = self.bn64(x)
#         x = self.LReLU(x)

#         for s1, s2, bn1, bn2 in self.blocks:
#             x = s1(x)
#             x = bn1(x)
#             x = self.LReLU(x)

#             x = s2(x)
#             x = bn2(x)
#             x = self.LReLU(x)

#         x = self.lin1(x.view(x.size()[0], -1))
#         x = self.LReLU(x)
#         x = self.lin2(x)
#         x = torch.sigmoid(x)

#         return x


