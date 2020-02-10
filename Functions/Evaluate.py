from glob import glob
import os
import numpy as np
from multiprocessing import Pool
from PIL import Image

import pydensecrf.densecrf as dcrf
eps = np.finfo(float).eps

def evaluate(param):
    pred_name, gt_name = param
    mask = Image.open(pred_name)
    gt = Image.open(gt_name)

    # print(mask.size, gt.size)

    mask = mask.resize(gt.size)
    mask = np.array(mask, dtype=np.float)
    mask = (mask - mask.min()) / (mask.max()-mask.min()+eps)

    gt = np.array(gt, dtype=np.uint8)

    if len(mask.shape) != 2:
        mask = mask[:, :, 0]

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    gt[gt != 0] = 1

    precision = []
    recall = []

    mae = np.abs(gt-mask).mean()

    binary = np.zeros(mask.shape)
    th = 2*mask.mean()
    if th >1:
        th = 1

    binary[mask >= th] = 1
    sb = (binary * gt).sum()

    pre = sb / (binary.sum() + eps)
    rec = sb / (gt.sum() + eps)

    thfm = 1.3 * pre * rec / (0.3 * pre + rec + eps)

    for th in np.linspace(0,1,21):
        binary = np.zeros(mask.shape)
        binary[mask >= th] = 1
        pre = (binary * gt).sum() / (binary.sum()+eps)
        rec = (binary * gt).sum() / (gt.sum()+eps)
        precision.append(pre)
        recall.append(rec)
    precision = np.array(precision)
    recall = np.array(recall)
    return thfm, mae, recall, precision



def fm_and_mae(pred_dir, gt_dir, output_dir=None, name=None):
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # pred_list = os.listdir(pred_dir)
    # gt_list = os.listdir(gt_dir)

    pred_list = sorted(glob(pred_dir + "/*"))
    gt_list = sorted(glob(gt_dir + "/*"))

    pool = Pool(4)
    results = pool.map(evaluate, zip(pred_list, gt_list))
    # print(results)
    thfm, meas, recs, pres = list(map(list, zip(*results)))
    m_mea = np.array(meas).mean()
    m_pres = np.array(pres).mean(0)
    m_recs = np.array(recs).mean(0)
    thfm = np.array(thfm).mean()

    fms = 1.3 * m_pres * m_recs / (0.3 * m_pres + m_recs + eps)
    maxfm = fms.max()
    meanfm = fms.mean()

    return maxfm, meanfm, m_mea, m_recs, m_pres

def f_measure(param):
    pres, recs = param
    return 1.3 * pres * recs / (0.3 * pres + recs + eps)