"""
Author: Yu Chen
Email: yu_chen2000@hust.edu.cn
"""


import torch
import torch.nn as nn


class Normal_encoder(nn.Module):
    def __init__(self, resize, in_channels, out_channels):
        super(Normal_encoder, self).__init__()
        # input 32 1 128 * 128 -> 32 14
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize = resize
        ##
        nn_size = self.calculate_size(self.resize, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        # nn_size must be integer
        assert (nn_size % 1 == 0)
        nn_size = int(nn_size)
        self.encoder_cnn_1 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features = 128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )
        self.encoder_cnn_2 = nn.Sequential(
            self.cnn_block(in_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 1, 4, 2, 1),
            nn.Flatten(),
            self.nn_block(in_features=nn_size * nn_size, out_features=128),
            nn.Dropout(),
            self.nn_block(in_features=128, out_features=7),
        )

    def calculate_size(self, pixel_size, kernel_size, stride, padding):
        size = (pixel_size + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
        return size

    def cnn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    def nn_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, Gain, phase):
        gain = self.encoder_cnn_1(Gain)
        phase = self.encoder_cnn_2(phase)
        y = torch.concat([gain, phase], dim=1)
        return y


class Normal_decoder(nn.Module):
    def __init__(self, resize, encoder_output_size, in_channels, out_channels):
        super(Normal_decoder, self).__init__()
        # 32 14 -> 32 1 128 128
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resize = resize
        ##
        nn_size = self.calculate_size(8, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        nn_size = self.calculate_size(nn_size, 4, 2, 1)
        # nn_size must be integer
        assert (nn_size == resize)
        nn_size = int(nn_size)

        self.decoder_cnn_1 = nn.Sequential(
            self.nn_block(encoder_output_size, 128),
            nn.Dropout(),
            self.nn_block(128, 8 * 8),
            nn.Unflatten(1, (1, 8, 8)),
            self.cnn_block(1, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            nn.ConvTranspose2d(2 * out_channels, in_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.decoder_cnn_2 = nn.Sequential(
            self.nn_block(encoder_output_size, 128),
            nn.Dropout(),
            self.nn_block(128, 8 * 8),
            nn.Unflatten(1, (1, 8, 8)),
            self.cnn_block(1, 2 * out_channels, 4, 2, 1),
            self.cnn_block(2 * out_channels, 4 * out_channels, 4, 2, 1),
            self.cnn_block(4 * out_channels, 2 * out_channels, 4, 2, 1),
            nn.ConvTranspose2d(2 * out_channels, in_channels, 4, 2, 1),
            nn.Tanh(),
        )


    def calculate_size(self, pixel_size, kernel_size, stride, padding):
        size = (pixel_size - 1) * stride - 2 * padding + 1 * (kernel_size - 1) + 1
        return size

    def cnn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
            ## nn.Sigmoid()
        )
    def nn_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
            ## nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        gain = self.decoder_cnn_1(x)
        phase = self.decoder_cnn_2(x)
        return (gain, phase)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def text_autoencoder():
    encoder = Normal_encoder(128, 1, 2)
    decoder = Normal_decoder(128, 14, 1, 2)
    a = torch.rand(size=(32,1,128,128))
    b = torch.rand(size=(32, 14))
    y = encoder(a, a)
    print(y.size())
    (gain, phase) = decoder(b)
    print(gain.size())
    print(phase.size())



if __name__ == '__main__':
    text_autoencoder()
