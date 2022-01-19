import sys
import logging
import daisy
import sklearn.metrics
import skimage.measure
import time

def calculate_variation_of_information(segmentation1, segmentation2):
    assert segmentation1.shape == segmentation2.shape, "segmentations should be same size"
    
    ret = skimage.measure.shannon_entropy(segmentation1)
    ret += skimage.measure.shannon_entropy(segmentation2)
    ret -= 2 * sklearn.metrics.mutual_info_score(segmentation1.flatten(), segmentation2.flatten())
    return ret

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

    start = time.time()
    voi = calculate_variation_of_information(gt,seg)
    print("time to calc shannon voi: ",time.time() - start)
    print(voi)
