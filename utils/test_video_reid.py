import os
import sys
import time
import random
import datetime
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


def print_time(string=''):
    ctime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    res = ctime + ' | ' + string
    print(res)

def extract_feat_sampled_frames(model, vids, cam, use_gpu=True):
    """
    :param model:
    :param vids: (b, 3, t, 256, 128)
    :param use_gpu:
    :return:
        features: (b, c)
    """
    if use_gpu:
        vids = vids.cuda()
        cam = cam.cuda()
    # print(vids.shape)
    # print(vids[0,0,:,0,:])
    # assert 1 < 0
    f = model(vids, cam_label=cam)    # (b, t, c)
    # f = f.mean(-1)
    f = f.data.cpu()
    return f

def extract_feat_all_frames(model, vids, max_clip_per_batch=45, use_gpu=True):
    """
    :param model:
    :param vids:    (_, b, c, t, h, w)
    :param max_clip_per_batch:
    :param use_gpu:
    :return:
        f, (1, C)
    """
    if use_gpu:
        vids = vids.cuda()

    b, c, t, h, w = vids.size()

    f = model(vids) # (b, t, c)
    # f = f.mean(-1)   # (b, c)

    # f = f.mean(0, keepdim=True)
    f = f.data.cpu()
    return f

def _feats_of_loader(model, loader, feat_func=extract_feat_sampled_frames, use_gpu=True):
    qf, q_pids, q_camids = [], [], []

    pd = tqdm(total=len(loader), ncols=120, leave=False)
    for batch_idx, (vids, pids, camids) in enumerate(loader):
        pd.update(1)

        f = feat_func(model, vids, camids, use_gpu=use_gpu)
        qf.append(f)
        q_pids.extend(pids.numpy())
        q_camids.extend(camids.numpy())
    pd.close()

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    return qf, q_pids, q_camids

def _eval_format_logger(cmc, mAP, ranks, desc=''):
    print_time("Results {}".format(desc))
    ptr = "mAP: {:.2%}".format(mAP)
    for r in ranks:
        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
    print_time(ptr)
    print_time("--------------------------------------")

def _cal_dist(qf, gf, distance='cosine'):
    """
    :param logger:
    :param qf:  (query_num, feat_dim)
    :param gf:  (gallery_num, feat_dim)
    :param distance:
         cosine
    :return:
        distance matrix with shape, (query_num, gallery_num)
    """
    if distance == 'cosine':
        qf = F.normalize(qf, dim=1, p=2)
        gf = F.normalize(gf, dim=1, p=2)
        distmat = -torch.matmul(qf, gf.transpose(0, 1))
    else:
        raise NotImplementedError
    return distmat


def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = torch.argsort(distmat, dim=1) # from small to large
    index = index.numpy()

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # ground truth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    # if num_no_gt > 0:
    #     print("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP

def test(model, queryloader, galleryloader, use_gpu, cfg, ranks=None):
    if ranks is None:
        ranks = [1, 5, 10, 20]
    since = time.time()
    if use_gpu:
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"
        model.to(device)
    model.eval()

    if cfg.TEST.ALL_FRAMES:
        print("use all frame")
        feat_func = extract_feat_all_frames
    else:
        feat_func = extract_feat_sampled_frames

    qf, q_pids, q_camids = _feats_of_loader(
        model,
        queryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = _feats_of_loader(
        model,
        galleryloader,
        feat_func,
        use_gpu=use_gpu)
    print_time("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    if cfg.DATASETS.NAMES == 'mars':
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    time_elapsed = time.time() - since
    print_time('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print_time("Computing distance matrix")
    # torch.save(qf, 'qf.pt')
    # torch.save(gf, 'gf.pt')
    # assert 1 < 0
    distmat = _cal_dist(qf=qf, gf=gf, distance=cfg.TEST.DISTANCE)
    print_time("Computing CMC and mAP")
    print(q_pids)
    print(distmat)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    _eval_format_logger(cmc, mAP, ranks, '')

    return cmc, mAP, ranks