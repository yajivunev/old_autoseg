# coding=utf-8

import numpy as np
import torch
import logging
import sys
import daisy
import time

# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py

def torch_rand(seg, gt, all_stats=False):
    # segA is truth, segB is query
    segA = torch.ravel(gt.to(torch.float))
    segB = torch.ravel(seg)
    n = segA.size()[0]

    n_labels_A = int((torch.amax(segA) + 1).item())
    n_labels_B = int((torch.amax(segB) + 1).item())
    indices = torch.stack((segA,segB))

    if torch.cuda.is_available():
        ones_data = torch.ones(n,dtype=torch.float).cuda()
        a_indices = torch.tensor(range(1,n_labels_A)).long().cuda()
        b_indices = torch.tensor(range(1,n_labels_B)).long().cuda()
        c_indices = torch.tensor([0]).long().cuda()
    else:
        ones_data = torch.ones(n,dtype=torch.float)
        a_indices = torch.tensor(range(1,n_labels_A)).long()
        b_indices = torch.tensor(range(1,n_labels_B)).long()
        c_indices = torch.tensor([0]).long()

    p_ij = torch.sparse_coo_tensor(indices,ones_data,(n_labels_A, n_labels_B),requires_grad=True)
    a = torch.index_select(p_ij,0,a_indices)
    b = torch.index_select(a,1,b_indices)
    c = torch.index_select(a,1,c_indices)
    d = b.multiply(b)

    a_i = torch.sparse.sum(a,1)
    b_i = torch.sparse.sum(b,0)

    sumA = torch.sparse.sum(a_i * a_i)
    sumB = torch.sparse.sum(b_i * b_i) + (torch.sparse.sum(c) / n)
    sumAB = torch.sparse.sum(d) + (torch.sparse.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore
    return are

def ds_wrapper(in_file, in_ds):

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    return ds

if __name__ == "__main__":

    gt_file = sys.argv[1]
    gt_dataset = sys.argv[2]

    seg_file = sys.argv[3]
    seg_dataset = sys.argv[4]

    gt = ds_wrapper(gt_file, gt_dataset)
    seg = ds_wrapper(seg_file, seg_dataset)

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    logging.info("Converting seg to nd array...")
    seg = seg.to_ndarray()

    time_start = time.time()
    are = torch_rand(gt[:10][::10,::10],seg[:10][::10,::10])
    print("time: ", time.time() - time_start)
    print(are)
