import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, conv_kernel_size, complex):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, complex, conv_kernel_size, padding=1),
            nn.Conv2d(complex, complex, conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex),
        )
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(complex, complex * 2, conv_kernel_size, padding=1),
            nn.Conv2d(complex * 2, complex * 2, conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * 2),
        )
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(complex * 2, complex * (2 ** 2), conv_kernel_size, padding=1),
            nn.Conv2d(complex * (2 ** 2), complex * (2 ** 2), conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * (2 ** 2)),
        )
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(complex * (2 ** 2), complex * (2 ** 3), conv_kernel_size, padding=1),
            nn.Conv2d(complex * (2 ** 3), complex * (2 ** 3), conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * (2 ** 3)),
        )
        self.mp4 = nn.MaxPool2d(2)

    def forward(self, x):
        block1_out = self.block1(x)
        mp1 = self.mp1(block1_out)
        block2_out = self.block2(mp1)
        mp2 = self.mp2(block2_out)
        block3_out = self.block3(mp2)
        mp3 = self.mp3(block3_out)
        block4_out = self.block4(mp3)
        mp4 = self.mp4(block4_out)
        skip_features = [block1_out, block2_out, block3_out, block4_out]
        return mp4, skip_features

class Decoder (nn.Module):
    def __init__(self, conv_kernel_size, complex):
        super(Decoder, self).__init__()
        self.upScale1 = nn.ConvTranspose2d(complex * (2 ** 4), complex * (2 ** 3), kernel_size=2, stride=2, padding=0, output_padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(complex * (2 ** 4), complex * (2 ** 3), conv_kernel_size, padding=1),
            nn.Conv2d(complex * (2 ** 3), complex * (2 ** 3), conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * (2 ** 3))
        )
        self.upScale2 = nn.ConvTranspose2d(complex * (2 ** 3), complex * (2 ** 2), kernel_size=2, stride=2, padding=0, output_padding=1)
        self.block2 = nn.Sequential(
            nn.Conv2d(complex * (2 ** 3), complex * (2 ** 2), conv_kernel_size, padding=1),
            nn.Conv2d(complex * (2 ** 2), complex * (2 ** 2), conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * (2 ** 2))
        )
        self.upScale3 = nn.ConvTranspose2d(complex * (2 ** 2), complex * 2, kernel_size=2, stride=2, padding=0)
        self.block3 = nn.Sequential(
            nn.Conv2d(complex * (2 ** 2), complex * 2, conv_kernel_size, padding=1),
            nn.Conv2d(complex * 2, complex * 2, conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex * 2)
        )
        self.upScale4 = nn.ConvTranspose2d(complex * 2, complex, kernel_size=2, stride=2, padding=0)
        self.block4 = nn.Sequential(
            nn.Conv2d(complex * 2, complex, conv_kernel_size, padding=1),
            nn.Conv2d(complex, complex, conv_kernel_size, padding=1),
            nn.BatchNorm2d(complex)
        )

    def forward(self, x, skip_features):
        up1 = self.upScale1(x)
        up1 = up1[:,:,:,:-1]
        skip1 = torch.cat((up1, skip_features[3]), dim=1)
        block1_out = self.block1(skip1)

        up2 = self.upScale2(block1_out)[:,:,:-1,:-1]

        skip2 = torch.cat((up2, skip_features[2]), dim=1)
        block2_out = self.block2(skip2)

        skip3 = torch.cat((self.upScale3(block2_out), skip_features[1]), dim=1)
        block3_out = self.block3(skip3)

        skip4 = torch.cat((self.upScale4(block3_out), skip_features[0]), dim=1)
        block4_out = self.block4(skip4)

        return block4_out
    
class Bottle(nn.Module):
    def __init__(self, conv_kernel_size, complex):
        super(Bottle, self).__init__()
        self.conv1 = nn.Conv2d(complex * (2 ** 3), complex * (2 ** 4), conv_kernel_size, padding=1)
        self.conv2 = nn.Conv2d(complex * (2 ** 4), complex * (2 ** 4), conv_kernel_size, padding=1)
        self.bn    = nn.BatchNorm2d(complex * (2 ** 4))

    def forward(self, x):
        conv1_features = self.conv1(x)
        conv2_features = self.conv2(conv1_features)
        bn_features = self.bn(conv2_features)
        return bn_features