import torch
import torch.nn as nn

import numpy as np
import glob

from .Progressive_Unet import *

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k_size=1, padd=0, stride=1):
        """
        :param in_ch: Size of input channel
        :param out_ch: Size of output channel
        :param k_size: Size of convolution kernel
        :param padd: type of padding
        :param stride: size of stride
        """

        super(ConvBlock, self).__init__()
        # self.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padd)),
        # self.add_module('norm', nn.BatchNorm2d(out_ch)),
        # self.add_module('ReLU', nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, residual=True):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        if residual:
            x2 = x1 + x2

        return x2


class DeConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, k_size=2, padd=0, stride=2):
        super(DeConvBlock, self).__init__()

        self.add_module('deconv', nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padd))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('ReLU', nn.ReLU(inplace=True))


class SkipUpScale(nn.Module):
    activation = None
    def __init__(self, in_ch, out_ch, skip_in=None, ref_model=None):
        super(SkipUpScale, self).__init__()

        self.deconv = DeConvBlock(in_ch, out_ch)
        self.conv = ConvBlock(out_ch + skip_in, out_ch, k_size=3, padd=1)

    def forward(self, x, x2):
        x = self.deconv(x)
        x = torch.cat([x, x2], 1)
        x = self.conv(x)

        return x


class OutputConv(nn.Module):
    def __init__(self, in_ch):
        super(OutputConv, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid((x))
        return x

class Hook():
    def __init__(self, module, backward=False):
        # if backward==False:
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        # else:
        #     self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()




