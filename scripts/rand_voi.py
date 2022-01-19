import daisy
import logging
import sys
import time
import numpy as np
from funlib.evaluate import rand_voi

logging.basicConfig(level=logging.INFO)

""" Script to evaluate clustering metrics (RAND,VOI,NVI,NID) between two 3d arrays. """

def compute_rand_voi(
        truth,
        test):

    logging.info("Calculating RAND,VOI,NVI,NID... \n")

    time_start = time.time()

    voi_report = rand_voi(truth, test, return_cluster_scores=False)
    
    print("time to compute RAND VOI: ", time.time() - time_start,"\n")
    metrics = voi_report.copy()

    rand_split = metrics['rand_split']
    rand_merge = metrics['rand_merge']
    rand = rand_split + rand_merge

    voi_split = metrics['voi_split']
    voi_merge = metrics['voi_merge']
    voi = voi_split + voi_merge

    nvi_split = metrics['nvi_split']
    nvi_merge = metrics['nvi_merge']
    nvi = nvi_split + nvi_merge

    nid = metrics['nid']

    logging.info(f"RAND: {rand}, RAND split: {rand_split}, RAND merge: {rand_merge}")
    logging.info(f"VOI: {voi}, VOI split: {voi_split}, VOI merge: {voi_merge}")
    logging.info(f"NVI: {nvi}, NVI split: {nvi_split}, NVI merge: {nvi_merge}")
    logging.info(f"NID: {nid}")

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

    compute_rand_voi(gt,seg)
