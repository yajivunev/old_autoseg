import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import logging
import numpy as np
import time
import os
import sys
from funlib.evaluate import rand_voi
from funlib.segment.arrays import replace_values

from multiprocessing import Process,Manager,Pool

""" Script to evaluate VOI,NVI,NID against ground truth for a fragments dataset at different 
agglomeration thresholds and find the best threshold. """

base_dir = "/scratch1/04101/vvenu/autoseg"

def evaluate_thresholds(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        roi_offset=None,
        roi_shape=None):

    start = time.time()

    results_file = os.path.join(fragments_file,'results.out')

    logging.basicConfig(
            filename=results_file,
            filemode='a',
            level=logging.INFO)

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
    
    # parallel process

    manager = Manager()
    metrics = manager.dict()

    metrics["voi"] = manager.dict()
    metrics["nvi"] = manager.dict()
    metrics["nid"] = manager.dict()
   
    with Pool(10) as pool:
        pool.starmap(evaluate,[(t,fragments,gt,fragments_file,edges_collection,metrics) for t in thresholds])
    #pool = []

    #for t in thresholds:

    #    p = Process(target=evaluate, args=(t,fragments,gt,fragments_file,edges_collection,metrics,))
    #    p.start()
    #    pool.append(p)

    #for p in pool: p.join()

    logging.info("Best VOI,NVI,NID and respective thresholds: \n")
    best_voi = metrics['voi'][min(metrics['voi'].keys())]
    best_nvi = metrics['nvi'][min(metrics['nvi'].keys())]
    best_nid = metrics['nid'][min(metrics['nid'].keys())]
    
    voi_thresh,voi_split,voi_merge = best_voi['threshold'],best_voi['voi_split'],best_voi['voi_merge']
    nvi_thresh,nvi_split,nvi_merge = best_nvi['threshold'],best_nvi['nvi_split'],best_nvi['nvi_merge']
    nid,nid_thresh = best_nid['nid'],best_nid['threshold']
    
    logging.info("VOI: threshold= {} , VOI= {}, VOI_split= {} , VOI_merge= {}".format(voi_thresh,voi_split+voi_merge,voi_split,voi_merge))
    logging.info("NVI: threshold= {} , NVI= {}, NVI_split= {} , NVI_merge= {}".format(nvi_thresh,nvi_split+nvi_merge,nvi_split,nvi_merge))
    logging.info("NID: threshold= {} , NID= {}".format(nid_thresh,nid))

    print("VOI: threshold= {} , VOI= {}, VOI_split= {} , VOI_merge= {}".format(voi_thresh,voi_split+voi_merge,voi_split,voi_merge))
    print("NVI: threshold= {} , NVI= {}, NVI_split= {} , NVI_merge= {}".format(nvi_thresh,nvi_split+nvi_merge,nvi_split,nvi_merge))
    print("NID: threshold= {} , NID= {}".format(nid_thresh,nid))
    print(f"Time to evaluate thresholds = {time.time() - start}")

def ds_wrapper(in_file, in_ds):

    try:
        ds = daisy.open_ds(in_file, in_ds)
    except:
        ds = daisy.open_ds(in_file, in_ds + '/s0')

    return ds

def evaluate(
        threshold,
        fragments,
        gt,
        fragments_file,
        edges_collection,
        metrics):
    
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

    metrics["voi"][voi['voi_split']+voi['voi_merge']] = voi # voi sum is the key to metrics["voi"]
    metrics["nvi"][voi['nvi_split']+voi['nvi_merge']] = voi # nvi sum is key for metrics["nvi"]
    metrics["nid"][voi['nid']] = voi # nid is key for nid dict

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

        logging.info("Threshold: {}".format(threshold))
        logging.info("VOI sum: {}    VOI split: {}    VOI merge: {}".format(metrics['voi_split']+metrics['voi_merge'],metrics['voi_split'],metrics['voi_merge']))
        logging.info("NVI sum: {}    NVI split: {}    NVI merge: {}".format(metrics['nvi_split']+metrics['nvi_merge'],metrics['nvi_split'],metrics['nvi_merge']))
        logging.info("NID: {}\n".format(metrics['nid']))
        
        return metrics

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
