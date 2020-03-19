import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from skimage import io
from skimage import color, morphology, filters
import os
import random
from skimage.transform import resize

import pydensecrf.densecrf as dcrf

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# parameter
EPSILON = 1e-3
tau = 1.5


def crf(img, anno, to_tensor=False):
    img = np.transpose(img, (1, 2, 0))
    anno = np.transpose(anno, (1, 2, 0))
    # img = img.copy(order='C')
    img = np.ascontiguousarray(img, dtype=np.uint8)

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    n_energy = -np.log((1.0 - anno + EPSILON)) / (tau * sigmoid(1 - anno))
    p_energy = -np.log(anno + EPSILON) / (tau * sigmoid(anno))

    U = np.zeros((2, img.shape[0] * img.shape[1]), dtype=np.float32)
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = np.expand_dims(infer[1, :].reshape(img.shape[:2]), 0)
    if to_tensor:
        res = torch.from_numpy(res).unsqueeze(0)

    res = np.transpose(res, (1, 2, 0))

    return res


def make_heatmap(img_dir):
    img_list = os.listdir(img_dir)
    mean = np.zeros((224, 224, 3))
    for i, v in enumerate(img_list):
        if i % 1000 == 0:
            print(i)
        img = plt.imread(img_dir + img_list[i])
        img = resize(img, (224, 224, 3))
        mean += img

    return mean