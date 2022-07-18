import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import daisy
import json
import logging
import numpy as np
import time
import os
import sys
from funlib.evaluate import rand_voi,detection_scores
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
        crop,
        edges_collection,
        thresholds_minmax,
        thresholds_step):

    start = time.time()
    
    if crop != "":
        fragments_file = os.path.join(fragments_file,os.path.basename(crop)[:-4]+'zarr')
        crop_path = os.path.join(fragments_file,'crop.json')
        with open(crop_path,"r") as f:
            crop = json.load(f)
        
        crop_name = crop["name"]
        crop_roi = daisy.Roi(crop["offset"],crop["shape"])

    else:
        crop_name = ""
        crop_roi = None

    results_file = os.path.join(fragments_file,"results.json") 
    
    # open fragments
    print("Reading fragments from %s" %fragments_file)

    fragments = ds_wrapper(fragments_file, fragments_dataset)

    print("Reading gt from %s" %gt_file)

    gt = ds_wrapper(gt_file, gt_dataset)

    print("fragments ROI is {}".format(fragments.roi))
    print("gt roi is {}".format(gt.roi))

    vs = gt.voxel_size

    if crop_roi:
        common_roi = crop_roi

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
        pool.starmap(evaluate,[(t,fragments,gt,fragments_file,crop_name,edges_collection,metrics) for t in thresholds])
   
    voi_sums = {metrics[x]['voi_sum']:x for x in thresholds}
    #nvi_sums = {metrics[x]['nvi_sum']:x for x in thresholds}
    #nids = {metrics[x]['nid']:x for x in thresholds}

    voi_thresh = voi_sums[sorted(voi_sums.keys())[0]]
    #nvi_thresh = nvi_sums[sorted(nvi_sums.keys())[0]]
    #nid_thresh = nids[sorted(nids.keys())[0]]

    metrics = dict(metrics)
    metrics['best_voi'] = metrics[voi_thresh]

    os.makedirs(os.path.dirname(results_file),exist_ok=True)

    with open(results_file,"w") as f:
        json.dump(metrics,f,indent=4)

    print(f"best VOI: threshold= {voi_thresh} , VOI= {metrics[voi_thresh]['voi_sum']}, VOI_split= {metrics[voi_thresh]['voi_split']} , VOI_merge= {metrics[voi_thresh]['voi_merge']}")
    #print(f"best NVI: threshold= {nvi_thresh} , NVI= {metrics[nvi_thresh]['nvi_sum']}, NVI_split= {metrics[nvi_thresh]['nvi_split']} , NVI_merge= {metrics[nvi_thresh]['nvi_merge']}")
    #print(f"best NID: threshold= {nid_thresh} , NID= {metrics[nid_thresh]['nid']}")
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
        crop_name,
        edges_collection,
        metrics):
    
    segment_ids = get_segmentation(
            fragments,
            fragments_file,
            edges_collection,
            threshold)

    results = evaluate_threshold(
            edges_collection,
            crop_name,
            segment_ids,
            gt,
            threshold)

    metrics[threshold] = results


def get_segmentation(
        fragments,
        fragments_file,
        edges_collection,
        threshold):

    #print("Loading fragment - segment lookup table for threshold %s..." %threshold)
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

    #print("Relabeling fragment ids with segment ids...")

    segment_ids = replace_values(fragments, fragment_segment_lut[0], fragment_segment_lut[1])

    return segment_ids


def evaluate_threshold(
        edges_collection,
        crop_name,
        test,
        truth,
        threshold):

        gt = truth.copy().astype(np.uint64)
        segment_ids = test.copy().astype(np.uint64)

        assert gt.shape == segment_ids.shape

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

        #mask to object of central voxel if crop_name is not empty
        if crop_name != "" and not crop_name.startswith('crop'): #i.e, if it is an object crop

            #get central voxel
            middle = [int(x/2) for x in gt.shape]
            gt_center = gt[middle[0],middle[1],middle[2]]
            seg_center = segment_ids[middle[0],middle[1],middle[2]]

            #mask out all but central segment
            gt[gt != gt_center] = 0
            segment_ids[segment_ids != seg_center] = 0

            gt[gt == gt_center] = 1
            segment_ids[segment_ids == seg_center] = 1

            vois = rand_voi(
                    gt,
                    segment_ids,
                    return_cluster_scores=False)

            scores = detection_scores(
                    gt,
                    segment_ids,
                    voxel_size=[50,2,2]) #lazy
            
            metrics[f"{crop_name}_voi_split"] = vois["voi_split"]
            metrics[f"{crop_name}_voi_merge"] = vois["voi_merge"]
            metrics[f"{crop_name}_voi_sum"] = vois["voi_split"] + vois["voi_merge"]
             
            metrics["com_distance"] = float(scores["avg_distance"])
            metrics["iou"] = float(scores["avg_iou"])

        return metrics

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    evaluate_thresholds(**config)
