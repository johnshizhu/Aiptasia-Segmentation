import torch
import torch.nn as nn
from layer import Encoder, Decoder


class UNet(nn.Module):
    def __init__(self, in_channels, conv_kernel_size):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, conv_kernel_size)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, conv_kernel_size),
            nn.Conv2d(1024, 1024, conv_kernel_size),
            nn.BatchNorm2d(1024)
        )
        self.decoder = Decoder(conv_kernel_size)
        self.last_conv = nn.Conv2d(64, 1, 1)
        self.last_sigm = nn.Sigmoid()

    def forward(self, x):
        encoder_features, skip_connections = self.encoder(x)
        bottle_features = self.bottleneck(encoder_features)
        decoder_features = self.decoder(bottle_features, skip_connections)
        final_conv_features = self.last_conv(decoder_features)
        final_sig_features = self.last_sigm(final_conv_features)
        return final_sig_features