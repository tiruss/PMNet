from Models.vgg import vgg19
import torch
import torch.nn as nn

network = vgg19(pretrained=True)

# print(network.features[0])

for i, v in enumerate(network.features):
    print(i, v)

nn.Sequential()