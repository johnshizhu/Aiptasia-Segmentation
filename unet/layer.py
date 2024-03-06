import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, in_channels, conv_kernel_size):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, conv_kernel_size),
            nn.Conv2d(64, 64, conv_kernel_size),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, conv_kernel_size),
            nn.Conv2d(128, 128, conv_kernel_size),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, conv_kernel_size),
            nn.Conv2d(256, 256, conv_kernel_size),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, conv_kernel_size),
            nn.Conv2d(512, 512, conv_kernel_size),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.skip_features = None

    def forward(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        self.skip_features = [block1_out, block2_out, block3_out, block4_out]
        return block4_out

class Decoder (nn.Module):
    def __init__(self, conv_kernel_size):
        super(Decoder, self).__init__()
        self.upScale1 = nn.ConvTranspose2d(1024, 512, 2)
        self.block1 = nn.Sequential(
            nn.Conv2d(1024, 512, conv_kernel_size),
            nn.Conv2d(512, 512, conv_kernel_size),
            nn.BatchNorm2d(512)
        )
        self.upScale2 = nn.ConvTranspose2d(512, 256, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 256, conv_kernel_size),
            nn.Conv2d(256, 256, conv_kernel_size),
            nn.BatchNorm2d(256)
        )
        self.upScale3 = nn.ConvTranspose2d(256, 128, 2)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 128, conv_kernel_size),
            nn.Conv2d(128, 128, conv_kernel_size),
            nn.BatchNorm2d(128)
        )
        self.upScale4 = nn.ConvTranspose2d(128, 64, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, conv_kernel_size),
            nn.Conv2d(64, 64, conv_kernel_size),
            nn.BatchNorm2d(64)
        )

    def forward(self, x, skip_features):
        skip1 = torch.cat((self.upScale1(x), skip_features[3]), dim=1)
        block1_out = self.block1(skip1)
        skip2 = torch.cat((self.upScale2(block1_out), skip_features[2]), dim=1)
        block2_out = self.block2(skip2)
        skip3 = torch.cat((self.upScale3(block2_out), skip_features[1]), dim=1)
        block3_out = self.block3(skip3)
        skip4 = torch.cat((self.upScale4(block3_out), skip_features[0]), dim=1)
        block4_out = self.block4(skip4)

        return block4_out