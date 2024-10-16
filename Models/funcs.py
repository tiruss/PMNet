import torch.nn as nn
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os

dim_dict = {
    'densenet169': [64, 128, 256, 640, 1664],
    'vgg16': [64, 128, 256, 512, 512],
    'mobilenet2': [32, 24, 32, 64, 1280],
    'resnet101': [64, 256, 512, 1024, 2048]
}


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


def fraze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.requires_grad=False

def visualize(img, gt, contour, pred, contour_pred, epoch, iter):
    image = np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0))
    ground_truth = np.squeeze(np.transpose(gt.detach().cpu().numpy()[0], (1, 2, 0)), -1)
    contour_gt = np.squeeze(np.transpose(contour.detach().cpu().numpy()[0], (1, 2, 0)), -1)
    output = np.squeeze(np.transpose(pred.detach().cpu().numpy()[0], (1, 2, 0)), -1)
    contour_output = np.squeeze(np.transpose(contour_pred.detach().cpu().numpy()[0], (1, 2, 0)), -1)

    os.makedirs('result', exist_ok=True)
    # plt.subplot(3, 2, 1)
    # plt.imshow(image)
    # plt.subplot(3, 2, 3)
    # plt.imshow(ground_truth, cmap='gray')
    # plt.subplot(3, 2, 4)
    # plt.imshow(output, cmap='gray')
    # plt.subplot(3, 2, 5)
    # plt.imshow(contour_gt, cmap='gray')
    # plt.subplot(3, 2, 6)
    # plt.imshow(contour_output, cmap='gray')
    # plt.show()
    plt.imsave("result/img_{}_{}.jpg".format(epoch, iter), image)
    plt.imsave("result/gt_{}_{}.jpg".format(epoch, iter), ground_truth, cmap='gray')
    plt.imsave("result/output_{}_{}.jpg".format(epoch, iter), output, cmap='gray')
    plt.imsave("result/contour_gt_{}_{}.jpg".format(epoch, iter), contour_gt, cmap='gray')
    plt.imsave("result/contour_output_{}_{}.jpg".format(epoch, iter), contour_output, cmap='gray')

