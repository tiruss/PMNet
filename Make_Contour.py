import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imsave
import argparse
from scipy import ndimage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Directory of dataset', default='HKU-IS')
    parser.add_argument('--contour_dir', type=str, help='Save directory of coutours', default='contour')

    args = parser.parse_args()

    mask_dir = os.path.join('Datasets', args.data_dir, 'gt')
    mask_list = os.listdir(mask_dir)
    data_dir = os.path.join('Datasets', args.data_dir)

    os.makedirs(data_dir, exist_ok=True)
    contour_dir = os.path.join(data_dir, args.contour_dir)
    os.makedirs(contour_dir, exist_ok=True)

    for i in mask_list:
        mask = plt.imread(os.path.join(mask_dir, i))
        mask_name = i.split('.')[0]
        dx = ndimage.sobel(mask, 0)
        dy = ndimage.sobel(mask, 1)
        mag = np.hypot(dx, dy)
        mag = mag/mag.max()

        plt.imsave(os.path.join(contour_dir, mask_name) + ".png", mag, cmap='gray')