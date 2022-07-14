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
from multiprocessing.managers import SharedMemoryManager

""" Script to evaluate VOI,NVI,NID against ground truth for a fragments dataset at different 
agglomeration thresholds and find the best threshold. """

base_dir = "/scratch1/04101/vvenu/autoseg"


def evaluate_thresholds(
        gt_file,
        gt_dataset,
        fragments_file,
        fragments_dataset,
        object_name,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        roi_offset=None,
        roi_shape=None):

    start = time.time()

    results_file = os.path.join(fragments_file,'results.json')

    if object_name is not None:
        gt_dataset = os.path.join('objects',object_name,gt_dataset,'s2')
        fragments_dataset = os.path.join(object_name,fragments_dataset)
        results_file = os.path.join(fragments_file,object_name,'results.json')

    # open fragments
    print("Reading fragments from %s" %fragments_file)

    fragments = ds_wrapper(fragments_file, fragments_dataset)

    print("Reading gt from %s" %gt_file)

    gt = ds_wrapper(gt_file, gt_dataset)

    print("fragments ROI is {}".format(fragments.roi))
    print("gt roi is {}".format(gt.roi))

    vs = gt.voxel_size

    if roi_offset:
        common_roi = daisy.Roi(roi_offset, roi_shape)

    else:
        common_roi = fragments.roi.intersect(gt.roi)

    print("common roi is {}".format(common_roi))
    # evaluate only where we have both fragments and GT
    print("Cropping fragments, mask, and GT to common ROI %s", common_roi)
    fragments = fragments[common_roi]
    gt = gt[common_roi]

    print("Converting fragments to nd array...")
    fragments = fragments.to_ndarray()

    print("Converting gt to nd array...")
    gt = gt.to_ndarray()

    thresholds = list(np.arange(
        thresholds_minmax[0],
        thresholds_minmax[1],
        thresholds_step))

    print("Evaluating thresholds...")
    
    # parallel process
    manager = Manager()
    metrics = manager.dict()

    for threshold in thresholds:
        metrics[threshold] = manager.dict()

    with Pool(16) as pool:
        pool.starmap(evaluate,[(t,fragments,gt,fragments_file,object_name,edges_collection,metrics) for t in thresholds])
   
    voi_sums = {metrics[x]['voi_sum']:x for x in thresholds}
    nvi_sums = {metrics[x]['nvi_sum']:x for x in thresholds}
    nids = {metrics[x]['nid']:x for x in thresholds}

    voi_thresh = voi_sums[sorted(voi_sums.keys())[0]]
    nvi_thresh = nvi_sums[sorted(nvi_sums.keys())[0]]
    nid_thresh = nids[sorted(nids.keys())[0]]

    metrics = dict(metrics)
    metrics['best_thresholds'] = list(set((voi_thresh,nvi_thresh,nid_thresh)))

    with open(results_file,"w") as f:
        json.dump(metrics,f,indent=4)

    print(f"best VOI: threshold= {voi_thresh} , VOI= {metrics[voi_thresh]['voi_sum']}, VOI_split= {metrics[voi_thresh]['voi_split']} , VOI_merge= {metrics[voi_thresh]['voi_merge']}")
    print(f"best NVI: threshold= {nvi_thresh} , NVI= {metrics[nvi_thresh]['nvi_sum']}, NVI_split= {metrics[nvi_thresh]['nvi_split']} , NVI_merge= {metrics[nvi_thresh]['nvi_merge']}")
    print(f"best NID: threshold= {nid_thresh} , NID= {metrics[nid_thresh]['nid']}")
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
        object_name,
        edges_collection,
        metrics):
    
    segment_ids = get_segmentation(
            fragments,
            fragments_file,
            object_name,
            edges_collection,
            threshold)

    results = evaluate_threshold(
            edges_collection,
            segment_ids,
            gt,
            threshold)

    metrics[threshold] = results


def get_segmentation(
        fragments,
        fragments_file,
        object_name,
        edges_collection,
        threshold):

    #print("Loading fragment - segment lookup table for threshold %s..." %threshold)
    fragment_segment_lut_dir = os.path.join(
            fragments_file,
            object_name,
            'luts',
            'fragment_segment')

    fragment_segment_lut_file = os.path.join(
            fragment_segment_lut_dir,
            'seg_%s_%d.npz' % (edges_collection, int(threshold*100)))

    fragment_segment_lut = np.load(
            fragment_segment_lut_file)['fragment_segment_lut']

    assert fragment_segment_lut.dtype == np.uint64

    #print("Relabeling fragment ids with segment ids...")

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

        #print("Storing VOI values for threshold %f in DB" %threshold)

        metrics['threshold'] = threshold
        metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
        metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']
        metrics['merge_function'] = edges_collection.strip('edges_')

        return metrics

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
