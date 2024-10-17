import argparse
import os

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Data import dataloader
from Models.Progressive_Unet import *
from Models.funcs import visualize
from Trainer import inter_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', default='Datasets/DUTS-TR/imgs/')
    parser.add_argument('--gt_dir', default='Datasets/DUTS-TR/gt/')
    parser.add_argument('--contour_dir', default='Datasets/DUTS-TR/contour/')
    parser.add_argument('--batch_size', default=12)
    parser.add_argument('--down_scale', default=5, type=int)
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--log_path', default='logs')
    parser.add_argument('--reload', default=False)
    parser.add_argument('--weight', default='Weights/param_200.pth')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.gpus > 1:
        model = nn.DataParallel(Progressive_Unet().cuda())
    else:
        model = Progressive_Unet().cuda()

    dataset = dataloader.custom_dataloader(img_dir=args.img_dir, mask_dir=args.gt_dir, contour_dir=args.contour_dir,
                                           train=True, down_scale=args.down_scale)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    writer = SummaryWriter(os.path.join('path', args.log_path))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    os.makedirs('Weights', exist_ok=True)

    if args.reload == False:
        mae = 0
        for epoch in range(0, 5):
            for i, batch in enumerate(data_loader):
                model.zero_grad()

                img = batch[0]['image'].to(device)
                gt = batch[0]['mask'].to(device)
                contour = batch[0]['contour'].to(device)

                pred, contour_pred = model(img)[0], model(img)[1]

                bce_loss = nn.BCELoss()
                mask_bce = bce_loss(pred, gt)
                contour_bce = bce_loss(contour_pred, contour)

                criterion = mask_bce + contour_bce

                criterion.backward()
                optimizer.step()

                mae = (torch.abs(pred - gt).sum() / dataset.__len__()) / 255.

                print('Epoch: {} Batch: {} / {} Mask Loss: {:.5f} / Contour Loss: {:.5f} /  MAE {:.5f}'.format(epoch + 1,len(data_loader),
                                                                                                               i, mask_bce.mean().item(),
                                                                                                               contour_bce.mean().item(), mae))

                if i != 0 and i % 100 == 0:
                    visualize(img, gt, contour, pred, contour_pred, epoch + 1, i)

                if i == len(data_loader)-1:
                    writer.add_scalar("training loss", criterion.mean().item(), epoch)

            if (epoch + 1) % 5 == 0:
                torch.save(model.module.state_dict(), "./Weights/param_{}.pth".format(epoch + 1))

        inter_net1 = inter_train(args, epochs=(6, 16), scale_ratio=1)
        inter_net2 = inter_train(args, epochs=(16, 31), scale_ratio=2)
        inter_net3 = inter_train(args, epochs=(31, 71), scale_ratio=3)
        inter_net4 = inter_train(args, epochs=(71, 121), scale_ratio=4)
        inter_net5 = inter_train(args, epochs=(121, 201), scale_ratio=5)
    else:
        net = inter_train(args, epochs=(201, 251), scale_ratio=5)
