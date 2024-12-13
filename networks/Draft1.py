import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################### Attention gate

class Attention_block(nn.Module):
    def __init__(self, g_channels, x_channels, mid_channels):
        super(Attention_block, self).__init__()
        self.g_channels = g_channels
        self.x_channels = x_channels
        self.mid_channels = mid_channels

        self.W_g = nn.Sequential(
            nn.Conv2d(self.g_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(self.x_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.mid_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(self.mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        diffY = x.size()[2] - g.size()[2]
        diffX = x.size()[3] - g.size()[3]
        g = F.pad(g, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi



'''
This is an attention U-Net example to check how I used the module in it.
"conv_block" and "up_conv" are just some convolutional layer blocks
'''


class Attention_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Attention_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(g_channels=512, x_channels=512, mid_channels=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(g_channels=256, x_channels=256, mid_channels=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(g_channels=128, x_channels=128, mid_channels=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(g_channels=64, x_channels=64, mid_channels=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
