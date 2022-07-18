import logging
from rand_voi import compute_rand_voi
import sys
import os
import daisy
import numpy as np
from skimage.metrics import adapted_rand_error

def ds_wrapper(in_file, in_ds):

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    return ds


if __name__ == "__main__":

    gt_file = sys.argv[1]
    gt_dataset = sys.argv[2]
    mask_dataset = sys.argv[3] #assuming mask dataset is in gt_file

    seg_file = sys.argv[4]
    seg_dataset = sys.argv[5]

    gt = ds_wrapper(gt_file, gt_dataset)
    seg = ds_wrapper(seg_file, seg_dataset)
    mask = ds_wrapper(gt_file, mask_dataset)

    roi = mask.roi

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray(roi)

    logging.info("Converting seg to nd array...")
    seg = seg.to_ndarray(roi).astype(np.uint64)

    logging.info("Convering mask to nd array...")
    mask = mask.to_ndarray(roi).astype(bool)

    #seg = np.pad(seg,((0,0),(20,20),(20,20)))

    seg = seg * mask #masking out unlabelled in seg
    gt = gt * mask #just in case

    compute_rand_voi(gt,seg)

    print(adapted_rand_error(image_true=gt,image_test=seg))
