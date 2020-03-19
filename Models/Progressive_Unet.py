import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from .resnet import *
from .vgg import *
from .densenet import *
from .funcs import *
from .Network import *

import sys
import numpy as np

dim_dict = {
    'densenet169': [64, 128, 256, 640, 1664],
    'vgg16': [64, 128, 256, 512, 512],
    'mobilenet2': [32, 24, 32, 64, 1280],
    'resnet101': [64, 256, 512, 1024, 2048]
}

class Progressive_Unet(nn.Module):
    def __init__(self, ch_en=32, scale=5, base='resnet50', prev_layer=None):
        super(Progressive_Unet, self).__init__()

        self.scale = scale
        self.base = base
        self.ch_en = ch_en
        self.init_model = Initial_Layer()

        self.decoder1_1 = SkipUpScale(self.ch_en * 64, self.ch_en * 16, skip_in = self.ch_en * 16)
        self.decoder1_2 = SkipUpScale(self.ch_en * 16, self.ch_en * 8, skip_in = self.ch_en * 8)
        self.decoder1_3 = SkipUpScale(self.ch_en * 8, self.ch_en * 4, skip_in = self.ch_en * 4)
        self.decoder1_4 = SkipUpScale(self.ch_en * 4, self.ch_en * 2, skip_in = self.ch_en * 2)
        self.decoder1_5 = SkipUpScale(self.ch_en * 2, self.ch_en, skip_in = self.ch_en)
        self.out1_1 = nn.Conv2d(self.ch_en * 16, 1, kernel_size=1, padding=0)
        self.out1_2 = nn.Conv2d(self.ch_en * 8, 1, kernel_size=1, padding=0)
        self.out1_3 = nn.Conv2d(self.ch_en * 4, 1, kernel_size=1, padding=0)
        self.out1_4 = nn.Conv2d(self.ch_en * 2, 1, kernel_size=1, padding=0)
        self.out1_5 = nn.Conv2d(self.ch_en, 1, kernel_size=1, padding=0)

        self.decoder2_1 = SkipUpScale(self.ch_en * 64, self.ch_en * 16, skip_in = self.ch_en * 16)
        self.decoder2_2 = SkipUpScale(self.ch_en * 16, self.ch_en * 8, skip_in = self.ch_en * 8)
        self.decoder2_3 = SkipUpScale(self.ch_en * 8, self.ch_en * 4, skip_in = self.ch_en * 4)
        self.decoder2_4 = SkipUpScale(self.ch_en * 4, self.ch_en * 2, skip_in = self.ch_en * 2)
        self.decoder2_5 = SkipUpScale(self.ch_en * 2, self.ch_en, skip_in = self.ch_en)
        self.out2_1 = nn.Conv2d(self.ch_en * 16, 1, kernel_size=1, padding=0)
        self.out2_2 = nn.Conv2d(self.ch_en * 8, 1, kernel_size=1, padding=0)
        self.out2_3 = nn.Conv2d(self.ch_en * 4, 1, kernel_size=1, padding=0)
        self.out2_4 = nn.Conv2d(self.ch_en * 2, 1, kernel_size=1, padding=0)
        self.out2_5 = nn.Conv2d(self.ch_en, 1, kernel_size=1, padding=0)

        self.orig = ConvBlock(3, 64, 3, padd=1)
        self.orig2 = ConvBlock(64, 32, 3, padd=1)

        if prev_layer is not None:
            self.prev = self.prev_model(prev_layer)

    def prev_model(self, layer):
        new_model = nn.Sequential()
        for i, (name, m) in enumerate(layer.named_children()):
            if not name == 'out':
                new_model.add_module(name, m)
                new_model[-1].load_state_dict(m.state_dict())

        return new_model


    def forward(self, x):

        orig = self.orig(x)
        orig = self.orig2(orig)

        x = self.init_model.scale1(x)
        s4 = self.init_model.skip4(x)
        x = self.init_model.scale2(x)
        s3 = self.init_model.skip3(x)
        x = self.init_model.scale3(x)
        s2 = self.init_model.skip2(x)
        x = self.init_model.scale4(x)
        s1 = self.init_model.skip1(x)
        x = self.init_model.scale5(x)
        o1 = self.init_model.out1(x)
        o2 = self.init_model.out2(x)

        if self.scale == 4 or self.scale == 3 or self.scale == 2 or self.scale == 1 or self.scale == 0:
            x1 = self.decoder1_1(x, s1)
            x2 = self.decoder2_1(x, s1)
            o1 = self.out1_1(x1)
            o2 = self.out2_1(x2)
            if self.scale == 3 or self.scale == 2 or self.scale == 1 or self.scale == 0:
                x1 = self.decoder1_2(x1, s2)
                x2 = self.decoder2_2(x2, s2)
                o1 = self.out1_2(x1)
                o2 = self.out2_2(x2)
                if self.scale == 2 or self.scale == 1 or self.scale == 0:
                    x1 = self.decoder1_3(x1, s3)
                    x2 = self.decoder2_3(x2, s3)
                    o1 = self.out1_3(x1)
                    o2 = self.out2_3(x2)
                    if self.scale == 1 or self.scale == 0:
                        x1 = self.decoder1_4(x1, s4)
                        x2 = self.decoder2_4(x2, s4)
                        o1 = self.out1_4(x1)
                        o2 = self.out2_4(x2)
                        if self.scale == 0:
                            x1 = self.decoder1_5(x1, orig)
                            x2 = self.decoder2_5(x2, orig)
                            o1 = self.out1_5(x1)
                            o2 = self.out2_5(x2)

        return [torch.sigmoid(o1), torch.sigmoid(o2)]

class Initial_Layer(nn.Module):
    def __init__(self):
        super(Initial_Layer, self).__init__()

        self.ch_en = 32

        base_model = resnet50(pretrained=True)
        base_model = nn.Sequential(*list(base_model.children())[:-2])  # Remove last layers

        self.scale1 = nn.Sequential(*base_model[:3])
        self.skip4 = ConvBlock(64, 64, k_size=1, padd=0)
        self.scale2 = nn.Sequential(*base_model[3:5])
        self.skip3 = ConvBlock(256, 128, k_size=1, padd=0)
        self.scale3 = base_model[5]
        self.skip2 = ConvBlock(512, 256, k_size=1, padd=0)
        self.scale4 = base_model[6]
        self.skip1 = ConvBlock(1024, 512, k_size=1, padd=0)
        self.scale5 = base_model[7]

        self.out1 = nn.Conv2d(self.ch_en * 64, 1, kernel_size=1, padding=0)
        self.out2 = nn.Conv2d(self.ch_en * 64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.scale1(x)
        x = self.scale2(x)
        x = self.scale3(x)
        x = self.scale4(x)
        x = self.scale5(x)

        x = self.out(x)



        return torch.sigmoid(x)











