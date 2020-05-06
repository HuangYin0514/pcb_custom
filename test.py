
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import distance
import utils
from dataloader import getDataLoader
from model import PCBModel

# ---------------------- Extract features ----------------------


def get_cam_label(img_path):
    camera_ids = []
    labels = []
    for path, _ in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return np.array(camera_ids), np.array(labels)


def extract_feature(model, inputs, requires_norm, vectorize, requires_grad=False):

    # Move to model's device
    inputs = inputs.to(next(model.parameters()).device)

    with torch.set_grad_enabled(requires_grad):
        features = model(inputs)

    size = features.shape

    if requires_norm:
        # [N, C*H]
        features = features.view(size[0], -1)

        # norm feature
        fnorm = features.norm(p=2, dim=1)
        features = features.div(fnorm.unsqueeze(dim=1))

    if vectorize:
        features = features.view(size[0], -1)
    else:
        # Back to [N, C, H=S]
        features = features.view(size)

    return features


# ---------------------- Evaluation ----------------------
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


# ---------------------- Start testing ----------------------
def test(model, dataset, dataset_path, batch_size, max_rank=50):
    model.eval()

    gallery_dataloader = getDataLoader(
        dataset, batch_size, dataset_path, 'gallery', shuffle=False, augment=False)
    query_dataloader = getDataLoader(
        dataset, batch_size, dataset_path, 'query', shuffle=False, augment=False)

    gallery_cams, gallery_labels = get_cam_label(
        gallery_dataloader.dataset.imgs)
    query_cams, query_labels = get_cam_label(query_dataloader.dataset.imgs)

    # Extract feature
    gallery_features = []
    query_features = []

    for inputs, _ in gallery_dataloader:
        gallery_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
    gallery_features = torch.cat(gallery_features, dim=0)

    for inputs, _ in query_dataloader:
        query_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
    query_features = torch.cat(query_features, dim=0)

    distmat = distance.compute_distance_matrix(
        query_features, gallery_features)
    all_cmc, mAP = eval_market1501(
        distmat, query_labels, gallery_labels, query_cams, gallery_cams, max_rank=max_rank)

    return all_cmc, mAP


if __name__ == "__main__":
    # from dataloader import getDataLoader
    # train_dataloader = getDataLoader(
    #     'market1501', 64, 'train', shuffle=True, augment=True)
    # gallery_dataloader = getDataLoader(
    #     'market1501', 4, 'gallery', shuffle=False, augment=False)
    # camera_ids, labels = get_cam_label(gallery_dataloader.dataset.imgs)
    from model import *
    model = build_model('PCB_p6', num_classes=len(train_dataloader.dataset.classes),
                        share_conv=True)
    CMC, mAP = test(model, 'market1501', 4)
    print('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' %
          (CMC[0], CMC[4], CMC[9], mAP))

    torch.cuda.empty_cache()
