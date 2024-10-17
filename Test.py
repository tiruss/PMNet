from Functions.Utils import crf
from Functions.Evaluate import fm_and_mae
from Models.Progressive_Unet import Progressive_Unet
from Data.dataloader import custom_dataloader
import pandas as pd

import glob
from PIL import Image
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="Weights/param_200.pth", type=str)
    parser.add_argument("--input_dir", default="DUTS-TE", type=str)
    parser.add_argument("--save_dir", default="Test_results")
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--result_save', default=True)
    parser.add_argument('--measure_test', default=True)
    parser.add_argument("--crf", default=False)

    args = parser.parse_args()

    device = torch.device('cuda')

    os.makedirs(args.save_dir, exist_ok=True)

    state_dict = torch.load(args.weight)
    model = Progressive_Unet(scale=0).to(device)
    model.load_state_dict(state_dict)

    dataset = custom_dataloader(img_dir=os.path.join(args.input_dir, "test_original/"),
                                mask_dir=os.path.join(args.input_dir, "test_label/"), train=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    os.makedirs(os.path.join(args.save_dir, args.input_dir), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.input_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.input_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.input_dir, 'mask_output'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.input_dir, 'contour_output'), exist_ok=True)

    model.eval()

    mae = 0
    for i, batch in enumerate(tqdm(dataloader)):
        img = batch[0]['image'].to(device)
        gt = batch[0]['mask'].to(device)
        name = batch[1]
        W, H = batch[2]

        with torch.no_grad():
            mask_pred = model(img)[0]
            contour_pred = model(img)[1]

        if mask_pred.size(0) != args.batch_size:
            args.batch_size = mask_pred.size(0)

        if args.result_save and args.crf == False:
            for j in range(args.batch_size):
                torchvision.utils.save_image(F.interpolate(contour_pred[j].unsqueeze_(0), size=[H[j].item(), W[j].item()]),
                                             os.path.join(args.save_dir, args.input_dir, 'contour_output', '{}'.format(name[j])))

                torchvision.utils.save_image(F.interpolate(mask_pred[j].unsqueeze_(0), size=[H[j].item(), W[j].item()]),
                                             os.path.join(args.save_dir, args.input_dir, 'mask_output', '{}'.format(name[j])))

        if args.result_save and args.crf == True:
            for j in range(args.batch_size):

                output = crf(img[j].detach().cpu().numpy(), mask_pred[j].detach().cpu().numpy())
                output = resize(output[:,:,0], (H[j].item(), W[j].item()))
                contour = resize(np.transpose(contour_pred[j].detach().cpu().numpy(), (1, 2, 0)), (H[j].item(), W[j].item()))
                # print(contour_pred.shape)
                plt.imsave(os.path.join(args.save_dir, args.input_dir, 'mask_output', '{}'.format(name[j])), output, cmap='gray')
                plt.imsave(os.path.join(args.save_dir, args.input_dir, 'contour_output', '{}'.format(name[j])), contour[:,:,0], cmap='gray')

        mae += torch.mean(torch.abs(mask_pred - gt))
    mae = mae / dataloader.__len__()

    if args.measure_test:
        maxfm, meanfm, m_mean, pres, recs, fms = fm_and_mae(pred_dir=os.path.join(args.save_dir, args.input_dir, "mask_output"),
                                                 gt_dir=os.path.join("Datasets/"  + args.input_dir, "gt"))

        print("Max F-measure: {:.4f}".format(maxfm))
        print("Mean F-measure: {:.4f}".format(meanfm))
        print("Mean Absolute Error: {:.4f}".format(m_mean))

        f = open("fm_dut-te.txt", "w")

        for i in fms[::-1]:
            f.write(str(i) + "\n")
        f.close()

        p = open("pr_dut-te.txt", "w")
        r = open("rc_dut-te.txt", "w")


        for i in pres[::-1]:
            p.write(str(i) + "\n")
        p.close()

        for i in recs[::-1]:
            r.write(str(i) + "\n")
        r.close()

