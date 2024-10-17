from Models.Progressive_Unet import Progressive_Unet
from Data import dataloader
from Models.funcs import visualize
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/0213')

def inter_train(args, epochs, scale_ratio, lr=0.001, prev_model=None):
    print("train {} * {}".format(scale_ratio*2*7, scale_ratio*2*7))

    model = nn.DataParallel(Progressive_Unet(scale=args.down_scale - scale_ratio).cuda())
    model.module.load_state_dict(torch.load(os.path.join('Weights', 'param_%s.pth' % str(epochs[0] - 1))))
    if args.reload:
        print("Reload {}".format(args.weight))
        model.module.load_state_dict(torch.load(args.weight))
    dataset = dataloader.custom_dataloader(img_dir=args.img_dir, mask_dir=args.gt_dir, contour_dir=args.contour_dir, train=True,
                                           down_scale=args.down_scale - scale_ratio)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs[0], epochs[1]):
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

            print('Epoch: {} Batch: {} / {} Mask Loss: {:.5f} / Contour Loss: {:.5f} /  MAE {:.5f}'.format(epoch,len(data_loader),
                                                                                                           i, mask_bce.mean().item(),
                                                                                                           contour_bce.mean().item(), mae))

            if i != 0 and i % 100 == 0:
                visualize(img, gt, contour, pred, contour_pred, epoch, i)

            if i == len(data_loader) - 1:
                writer.add_scalar("training loss", criterion.mean().item(), epoch)

        if epoch % (epochs[1] - 1) == 0:
            torch.save(model.module.state_dict(), "./Weights/param_{}.pth".format(epoch))

    return model