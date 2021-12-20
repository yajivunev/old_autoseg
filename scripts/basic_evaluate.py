import daisy
import logging
import sys
import numpy as np
from funlib.evaluate import rand_voi

logging.basicConfig(level=logging.INFO)

""" Script to evaluate clustering metrics (RAND,VOI,NVI,NID) between two 3d arrays. """

def evaluate(
        truth,
        test):

    logging.info("Evaluating... \n")

    voi_report = rand_voi(truth, test, return_cluster_scores=False)
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


    logging.info("RAND: %s", rand)
    logging.info("VOI: %s", voi)
    logging.info("NVI: %s", nvi)
    logging.info("NID: %s", nid)


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

    evaluate(gt,seg)
