import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from skimage import io
from skimage import color, morphology, filters
import os
import random

def move_to_gpu(x):
    if (torch.cuda.is_available()):
        x = x.to(torch.device('cuda'))
    return x

def move_to_cpu(x):
    x = x.to(torch.device('cpu'))
    return x

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3,2,0,1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None, None]
        x = x.transpose(3,2,0,1)
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor)

    return x

def read_image(opt):
    x = io.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x
