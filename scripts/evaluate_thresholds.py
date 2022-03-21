import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import logging
import lsd
import numpy as np
import time
import os
import sys
import waterz
from funlib.evaluate import rand_voi
from funlib.segment.arrays import replace_values
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

""" Script to evaluate VOI,NVI,NID against ground truth for a fragments dataset at different 
agglomeration thresholds and find the best threshold. """

base_dir = "/scratch1/04101/vvenu/autoseg"

def evaluate(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        roi_offset=None,
        roi_shape=None):

    logging.basicConfig(level=logging.INFO)

    # open fragments
    logging.info("Reading fragments from %s" %fragments_file)
    print("Reading fragments from %s" %fragments_file)

    fragments = ds_wrapper(fragments_file, fragments_dataset)

    logging.info("Reading gt from %s" %gt_file)
    print("Reading gt from %s" %gt_file)

    gt = ds_wrapper(gt_file, gt_dataset)

    logging.info("fragments ROI is {}".format(fragments.roi))
    logging.info("gt roi is {}".format(gt.roi))

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = fragments.roi.intersect(gt.roi)

    logging.info("common roi is {}".format(common_roi))
    # evaluate only where we have both fragments and GT
    logging.info("Cropping fragments and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    logging.info("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    logging.info("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    logging.info("Evaluating thresholds...")
    voi_dict = {}
    nvi_dict = {}
    nid_dict = {}
    for threshold in thresholds:
    
        segment_ids = get_segmentation(
                fragments,
                fragments_file,
                edges_collection,
                threshold)

        voi = evaluate_threshold(
                edges_collection,
                segment_ids,
                gt,
                threshold)
        voi_dict[voi['voi_split']+voi['voi_merge']] = voi # voi sum is the key to voi_dict
        nvi_dict[voi['nvi_split']+voi['nvi_merge']] = voi # nvi sum is key for nvi_dict
        nid_dict[voi['nid']] = voi # nid is key for nid dict

    logging.info("Best VOI,NVI,NID and respective thresholds: \n")
    best_voi = voi_dict[min(voi_dict.keys())]
    best_nvi = nvi_dict[min(nvi_dict.keys())]
    best_nid = nid_dict[min(nid_dict.keys())]
    
    voi_thresh,voi_split,voi_merge = best_voi['threshold'],best_voi['voi_split'],best_voi['voi_merge']
    nvi_thresh,nvi_split,nvi_merge = best_nvi['threshold'],best_nvi['nvi_split'],best_nvi['nvi_merge']
    nid_thresh = best_nid['threshold']
    
    print(" ")
    logging.info("VOI: threshold= {} , VOI= {}, VOI_split= {} , VOI_merge= {}".format(voi_thresh,voi_split+voi_merge,voi_split,voi_merge))
    logging.info("NVI: threshold= {} , NVI= {}, NVI_split= {} , NVI_merge= {}".format(nvi_thresh,nvi_split+nvi_merge,nvi_split,nvi_merge))
    logging.info("NID: threshold= {} , NID= {}".format(nid_thresh,best_nid['nid']))

def ds_wrapper(in_file, in_ds):

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    return ds

def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold):

    #logging.info("Loading fragment - segment lookup table for threshold %s..." %threshold)
    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment')

    fragment_segment_lut_file = os.path.join(
            fragment_segment_lut_dir,
            'seg_%s_%d.npz' % (edges_collection, int(threshold*100)))

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    #logging.info("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids

def evaluate_threshold(
        edges_collection,
        segment_ids,
        gt,
        threshold):

        #get VOI and RAND
        logging.info("Calculating VOI scores for threshold %f...", threshold)

        #logging.info(type(segment_ids))

        rand_voi_report = rand_voi(
                gt,
                segment_ids,
                return_cluster_scores=False)

        metrics = rand_voi_report.copy()

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        #logging.info("Storing VOI values for threshold %f in DB" %threshold)

        metrics['threshold'] = threshold
        metrics['merge_function'] = edges_collection.strip('edges_')

        logging.info("VOI sum: {}    VOI split: {}    VOI merge: {}".format(metrics['voi_split']+metrics['voi_merge'],metrics['voi_split'],metrics['voi_merge']))
        logging.info("NVI sum: {}    NVI split: {}    NVI merge: {}".format(metrics['nvi_split']+metrics['nvi_merge'],metrics['nvi_split'],metrics['nvi_merge']))
        logging.info("NID: {}\n".format(metrics['nid']))
        
        return metrics

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate(**config)
