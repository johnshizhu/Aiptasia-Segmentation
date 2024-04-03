import torch.nn as nn
from .layer import Encoder, Decoder, Bottle

class UNet(nn.Module):
    def __init__(self, in_channels, conv_kernel_size, complex):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, conv_kernel_size, complex)
        self.bottleneck = Bottle(conv_kernel_size, complex)
        self.decoder = Decoder(conv_kernel_size, complex)
        self.last_conv = nn.Conv2d(complex, 1, 1)
        self.last_sigm = nn.Sigmoid()

    def forward(self, x):
        encoder_features, skip_connections = self.encoder(x)
        bottle_features = self.bottleneck(encoder_features)
        decoder_features = self.decoder(bottle_features, skip_connections)
        final_conv_features = self.last_conv(decoder_features)
        final_sig_features = self.last_sigm(final_conv_features)
        return final_sig_features