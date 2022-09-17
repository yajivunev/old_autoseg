import sys
import json
import numpy as np
import logging
import itertools
import time

import tqdm
import daisy
from multiprocessing import Pool

import gc

from funlib.geometry import Roi,Coordinate
from funlib.evaluate import rand_voi

from scipy.ndimage import maximum_filter
from skimage.morphology import disk
from skimage.transform import rescale, downscale_local_mean
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

from lsd.post.fragments import watershed_from_affinities

import waterz


def run_val(args):

    raw_container = args['raw_container']
    pred_container = args['pred_container']
    roi = args['roi']
    downsampling_mode = args['downsampling_mode']
    factor = args['factor']
    target_dwt = args['target_dwt']
    components = args['components']
    normalize_lsds = args['normalize_lsds']
    affs_nb = args['affs_nb']
    affs_max_dist = args['affs_max_dist']
    affs_max_filter = args['affs_max_filter']
    fragments_dwt = args['fragments_dwt']
    bg_mask = args['bg_mask']
    merge_function = args['merge_function']

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }
    start_time = time.time()

    merge_function = waterz_merge_function[merge_function]
    
    #open datasets
    labels = daisy.open_ds(raw_container,"labels/s1")
    lsds = daisy.open_ds(pred_container,"lsds")

    if roi is None:
        roi = labels.roi
    else:
        roi = daisy.Roi(labels.roi.offset+daisy.Coordinate(roi[0]),roi[1])

    #load arrays
    labels = labels.to_ndarray(roi)
    lsds = lsds.to_ndarray(roi)

    #denoise lsds or convert to float32
    if target_dwt is not None:
        if target_dwt[0] == "tv":
            lsds = np.stack([denoise_tv_chambolle(x,weight=target_dwt[1]) for x in lsds])
            lsds = lsds.astype(np.float32)

        elif target_dwt[0] == "bilateral":
            lsds = np.stack([denoise_bilateral(lsds[:,z],sigma_color=target_dwt[1],channel_axis=0) for z in range(lsds.shape[1])],axis=1)
            lsds = lsds.astype(np.float32)
            
        else:
            raise KeyError("unknown denoising mode for target lsds")
            
    else: 
        lsds = (lsds/255.0).astype(np.float32) if lsds.dtype == np.uint8 else lsds
        
    #downsample lsds
    if factor > 1:
        if downsampling_mode == 'rescale':
            lsds = rescale(lsds,[1,1,1/factor,1/factor],anti_aliasing=True,order=1).astype(np.float32)
        elif downsampling_mode == 'local_mean':
            lsds = downscale_local_mean(lsds,(1,1,factor,factor))
            lsds = lsds.astype(np.float32)
        else:
            lsds = lsds[:,:,::factor,::factor]
        
    #get components of lsds
    if components is not None:
        
        comps = []
        
        for x in components:
            comps += [int(x)]

    #normalize lsds back to [0,1]
    if normalize_lsds:
        assert lsds.dtype==np.float32
        max_v = np.max(lsds)
        min_v = np.min(lsds)
        assert max_v <= 1.0
        assert min_v >= 0.0

        lsds = (lsds - min_v)/(max_v - min_v)

    #make affs
    z = np.linalg.norm(lsds[:,affs_nb:,:,:] - lsds[:,:-affs_nb,:,:], axis=0)
    y = np.linalg.norm(lsds[:,:,affs_nb:,:] - lsds[:,:,:-affs_nb,:], axis=0)
    x = np.linalg.norm(lsds[:,:,:,affs_nb:] - lsds[:,:,:,:-affs_nb], axis=0)
    
    shape = lsds.shape[1:]
    
    z = z[0:shape[0]-affs_max_dist,0:shape[1]-affs_max_dist,0:shape[2]-affs_max_dist]
    y = y[0:shape[0]-affs_max_dist,0:shape[1]-affs_max_dist,0:shape[2]-affs_max_dist]
    x = x[0:shape[0]-affs_max_dist,0:shape[1]-affs_max_dist,0:shape[2]-affs_max_dist]
    
    footprint = np.stack([np.zeros((3,3)),disk(1),np.zeros((3,3))])
    
    if affs_max_filter:
        z = maximum_filter(z,footprint=footprint)
        y = maximum_filter(y,footprint=footprint)
        x = maximum_filter(x,footprint=footprint)
    
    affs = 1 - np.stack([z,y,x])

    for c in range(3):
        max_v = np.max(affs[c])
        min_v = np.min(affs[c])
        
        affs[c] = (affs[c] - min_v)/(max_v - min_v)
    
    #remove lsds from memory
    del lsds
    gc.collect()

    #make fragments
    frags_time = time.time()
    frags = watershed_from_affinities(affs,denoise_wt=fragments_dwt,background_mask=bg_mask,fragments_in_xy=True,min_seed_distance=10)
    frags_time = time.time() - frags_time

    #crop labels and raw for eval,vis
    labels = labels[:-affs_max_dist,:-affs_max_dist*factor,:-affs_max_dist*factor]

    #agglomerate
    thresholds = [round(x,2) for x in np.arange(0,1,1/25)]

    generator = waterz.agglomerate(
            affs=affs,
            thresholds=thresholds,
            fragments=frags[0],
            scoring_function=merge_function,
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False)

    #over thresholds
    results = {}
    for thresh,seg in zip(thresholds,generator):
        
        seg = rescale(seg, [1,factor,factor], order=0);
        
        #eval
        eval_time = time.time()
        
        metrics = rand_voi(
            labels,
            seg,
            return_cluster_scores=False)

        eval_time = time.time() - eval_time

        metrics['merge_threshold'] = thresh
        metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
        metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']
        metrics['eval_time'] = eval_time

        for k in {'voi_split_i', 'voi_merge_j'}:
            del metrics[k]

        results[thresh] = metrics

    #rescale fragments
    if factor > 1:
            frags = rescale(frags[0], [1,factor,factor], order=0)
    else: frags = frags[0]

    #evaluate fragments
    assert labels.shape == frags.shape

    frags_metrics = rand_voi(
            labels,
            frags,
            return_cluster_scores=False)

    frags_metrics['voi_sum'] = frags_metrics['voi_split']+frags_metrics['voi_merge']
    frags_metrics['nvi_sum'] = frags_metrics['nvi_split']+frags_metrics['nvi_merge']

    for k in {'voi_split_i', 'voi_merge_j'}:
        del frags_metrics[k]
            
    frags_metrics['frags_time'] = frags_time

    #finish
    best_thresh = sorted([(results[x]['nvi_sum'],results[x]['merge_threshold']) for x in results])[0][1]

    results['frags'] = frags_metrics
    results['total_time'] = time.time() - start_time
    results['best'] = results[best_thresh]

    #clean up
    del generator
    del frags
    del labels
    del seg
    del affs
    gc.collect()

    return args | results["best"]


if __name__ == "__main__":

    grid = sys.argv[1]
    results_out = sys.argv[2]

    with open(grid,"r") as f:
        grid = json.load(f)

    keys, values = zip(*grid.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    length = len(arguments)

    #arguments = arguments[:2]

    results = {}

    print(f"total number of validation runs: {length}")
    
    with Pool(50) as pool:

        for i,result in enumerate(tqdm.tqdm(pool.imap_unordered(run_val,arguments,chunksize=1),total=length)):
                results[i] = result

    if results != {}:
        with open(results_out,'w') as f:
            json.dump(results,f,indent=4)
