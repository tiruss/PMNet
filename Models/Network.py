import torch
import torch.nn as nn

import numpy as np
import glob

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k_size, padd, stride):
        """
        :param in_ch: Size of input channel
        :param out_ch: Size of output channel
        :param k_size: Size of convolution kernel
        :param padd: type of padding
        :param stride: size of stride
        """

        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_ch)),
        self.add_module('ReLU', nn.ReLU(inplace=True))

class DeConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k_size, padd, stride):
        super(DeConvBlock, self).__init__()

        self.add_module('deconv', nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padd))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('ReLU', nn.ReLU(inplace=True))

class ResidualBlock(nn.Module):
    def __init__(self, opt):
        super(ResidualBlock, self).__init__()
        pass






