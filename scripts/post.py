import sys
import os
import json
import numpy as np
import daisy
import logging
import glob
import itertools
import time
import tqdm
from multiprocessing import Pool

import gc

from funlib.geometry import Roi,Coordinate
from funlib.evaluate import rand_voi

from scipy.ndimage import maximum_filter
from skimage.morphology import disk, ball
from skimage.transform import rescale, downscale_local_mean
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

from lsd.post.fragments import watershed_from_affinities

import waterz

from affogato.segmentation import compute_mws_segmentation


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


""" Script to perform post-processing on predicted affs,lsds,LR affs,or MWS affs, and evaluate. """


def post_affs(
        raw_container,
        pred_container,
        roi,
        factor,
        denoise_wt):

    total_time = time.time()

    labels = daisy.open_ds(raw_container,"labels/s1")
    affs = daisy.open_ds(pred_container,"affs")

    if roi is None:
        roi = labels.roi
    else:
        roi = daisy.Roi(labels.roi.offset+daisy.Coordinate(roi[0]),roi[1])

    #load arrays
    labels = labels.to_ndarray(roi)
    affs = affs.to_ndarray(roi)

    #denoise
    if denoise_wt is not None:
        affs = np.stack([denoise_tv_chambolle(x,weight=denoise_wt) for x in affs])
        affs = affs.astype(np.float32)

    #downsample
    if factor > 1:
        affs = rescale(affs,[1,1,1/factor,1/factor],anti_aliasing=True,order=1).astype(np.float32)

    #make fragments
    frags_time = time.time()
    frags = watershed_from_affinities(affs,denoise_wt=None,background_mask=False,fragments_in_xy=True)
    frags_time = time.time() - frags_time

    #relabel fragments for agglomeration
    
    #fragments_relabelled, n, fragment_relabel_map = relabel(
    #        frags[0],
    #        return_backwards_map=True)

    thresholds = [round(x,2) for x in np.arange(0,1,1/50)]

    #agglomeration
    seg_gen = waterz.agglomerate(
            affs=affs[:3],
            thresholds=thresholds,
            fragments=frags[0],
            scoring_function='OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False)

    results = {}
    for thresh,seg in zip(thresholds,seg_gen):

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
    frags = rescale(frags[0], [1,factor,factor], order=0)

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

    best_thresh = sorted([(results[x]['nvi_sum'],results[x]['merge_threshold']) for x in results])[0][1]

    #finish
    results['frags'] = frags_metrics
    results['total_time'] = time.time() - total_time
    results['best'] = results[best_thresh]

    return results


def get_affs(lsds, distance, max_distance, max_filter=True):
    
    z_distances = np.linalg.norm(lsds[:,distance:,:,:] - lsds[:,:-distance,:,:], axis=0)
    y_distances = np.linalg.norm(lsds[:,:,distance:,:] - lsds[:,:,:-distance,:], axis=0)
    x_distances = np.linalg.norm(lsds[:,:,:,distance:] - lsds[:,:,:,:-distance], axis=0)
    
    shape = lsds.shape[1:]
    
    z = z_distances[0:shape[0]-max_distance,0:shape[1]-max_distance,0:shape[2]-max_distance]
    y = y_distances[0:shape[0]-max_distance,0:shape[1]-max_distance,0:shape[2]-max_distance]
    x = x_distances[0:shape[0]-max_distance,0:shape[1]-max_distance,0:shape[2]-max_distance]
    
    footprint = np.stack([np.zeros((3,3)),disk(1),np.zeros((3,3))])
    
    if max_filter:
        z = maximum_filter(z,footprint=footprint)
        y = maximum_filter(y,footprint=footprint)
        x = maximum_filter(x,footprint=footprint)
    
    affs = 1 - np.stack([z,y,x])

    for c in range(3):
        max_v = np.max(affs[c])
        min_v = np.min(affs[c])
        
        affs[c] = (affs[c] - min_v)/(max_v - min_v)
    
    return affs


def post_lsds(
        raw_container,
        pred_container,
        roi,
        downsampling_mode,
        factor,
        target_dwt,
        components,
        normalize_lsds,
        affs_nb,
        affs_max_dist,
        affs_max_filter,
        fragments_dwt,
        bg_mask,
        merge_function):

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

        lsds = lsds[comps]

    #normalize lsds back to [0,1]
    if normalize_lsds:
        assert lsds.dtype==np.float32
        max_v = np.max(lsds)
        min_v = np.min(lsds)
        assert max_v <= 1.0
        assert min_v >= 0.0

        lsds = (lsds - min_v)/(max_v - min_v)

    #make affs
    affs = get_affs(
            lsds,
            affs_nb,
            affs_max_dist,
            affs_max_filter)

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
    del frags
    del labels
    del seg
    del affs
    gc.collect()

    return results


def post_mws(
        raw_container,
        pred_container,
        roi,
        factor,
        denoise_wt):
    
    total_time = time.time()

    labels = daisy.open_ds(raw_container,"labels/s1")
    affs = daisy.open_ds(pred_container,"affs")

    if roi is None:
        roi = labels.roi
    else:
        roi = daisy.Roi(labels.roi.offset+daisy.Coordinate(roi[0]),roi[1])

    #load arrays
    labels = labels.to_ndarray(roi)
    affs = affs.to_ndarray(roi)

    #denoise
    if denoise_wt is not None:
        affs = np.stack([denoise_tv_chambolle(x,weight=denoise_wt) for x in affs])
        affs = affs.astype(np.float32)

    #downsample
    if factor > 1:
        affs = rescale(affs,[1,1,1/factor,1/factor],anti_aliasing=True,order=1).astype(np.float32)

    if affs.shape[0] == 3: #direct nbhd
        neighborhood = np.array([

            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],

            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],

            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],

        ])
        
        affs = np.concatenate([affs,affs,affs,affs],axis=0)
        assert affs.shape[0] == 12
        
    elif affs.shape[0] == 12: #lr nbhd
        neighborhood = np.array([

            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],

            [-2, 0, 0],
            [0, -3, 0],
            [0, 0, -3],

            [-3, 0, 0],
            [0, -9, 0],
            [0, 0, -9],
            
            [-4, 0, 0],
            [0, -27, 0],
            [0, 0, -27]
        ])

    elif affs.shape[0] == 18: #lr nbhd + diagonals
        neighborhood = np.array([

            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],

            [-2, 0, 0],
            [0, -3, 0],
            [0, 0, -3],

            [-3, 0, 0],
            
            [-4, 0, 0],
            [0, -27, 0],
            [0, 0, -27]
        ])

        pos_diag = np.round(-9 * np.sin(np.linspace(0,np.pi,num=8,endpoint=False)))
        neg_diag = np.round(-9 * np.cos(np.linspace(0,np.pi,num=8,endpoint=False)))
        stacked_diag = np.stack([0*pos_diag,pos_diag,neg_diag],axis=-1)
        neighborhood = np.concatenate([neighborhood, stacked_diag]).astype(np.int8)
        
        assert len(neighborhood)==18

    elif affs.shape[0]==20: #mws paper nbhd
        neighborhood = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            # direct 3d nhood for attractive edges
            [-1, -1, -1],
            [-1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            # indirect 3d nhood for dam edges 
            [-2, 0, 0],
            [0, -3, 0],
            [0, 0, -3],
            [0, -9, 0],
            [0, 0, -9],
            # long range direct hood
            [0, -9, -9],
            [0, 9, -9],
            [0, -9, -4],
            [0, -4, -9],
            [0, 4, -9],
            [0, 9, -4],
            # inplane diagonal dam edges
            [0, -27, 0],
            [0, 0, -27]
            ])

    affs = 1 - affs
    sep = 3

    affs[:sep] = (affs[:sep] * -1)
    affs[:sep] = (affs[:sep] + 1)

    seg = compute_mws_segmentation(
            affs,
            neighborhood,
            sep,
            strides=[1, 10, 10])

    seg = rescale(seg, [1, factor, factor], order=0, preserve_range=True, anti_aliasing=False)

    #eval
    eval_time = time.time()

    metrics = rand_voi(
        labels,
        seg,
        return_cluster_scores=False)

    eval_time = time.time() - eval_time

    metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']
    metrics['eval_time'] = eval_time
    metrics['total_time'] = time.time() - total_time

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    return metrics


def eval_run(args):

    raw_container = args['raw_container']
    pred_container = args['pred_container']
    roi = args['roi']
    factor = args['factor']
    denoise_wt = args['denoise_wt']
        
    result = {'pred_container':pred_container}
    preds = os.listdir(pred_container)
    print(preds)

    if 'affs' in preds:

        print("doing affs")
        result['affs'] = post_affs(
                raw_container,
                pred_container,
                roi,
                factor,
                denoise_wt)

        print("doing mws")
        result['mws'] = post_mws(
                raw_container,
                pred_container,
                roi,
                factor,
                denoise_wt)

    if 'lsds' in preds:

        print("doing lsds")
        result['lsds'] = post_lsds(
                raw_container,
                pred_container,
                roi,
                downsampling_mode="local_mean",
                factor=1,
                target_dwt=["tv",0.1],
                components="0129",
                normalize_lsds=True,
                affs_nb=1,
                affs_max_dist=10,
                affs_max_filter=False,
                fragments_dwt=None,
                bg_mask=False)


    return result


if __name__ == "__main__":
    
    setup_path = sys.argv[1]
    raw_container = sys.argv[2]

    roi = [[500,400,200],[2500,2400,2400]]
    factor = 4
    denoise_wt = 0.05

    volume = os.path.basename(raw_container)

    pred_containers = glob.glob(os.path.join(setup_path,"*","50000",volume))
    print(pred_containers)

    n_runs = len(pred_containers)

    results = []

    arguments = {
            'raw_container':[raw_container],
            'pred_container':pred_containers,
            'roi':[roi],
            'factor':[factor],
            'denoise_wt':[denoise_wt]}

    keys, values = zip(*arguments.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(arguments)

    with Pool(n_runs) as pool:

        for result in tqdm.tqdm(pool.imap(eval_run,arguments), total=len(arguments)):
                results.append(result)
   
    results = {'results': results}

    available_preds = os.listdir(pred_containers[0])
    print(available_preds)

    #average runs
    print("averaging runs")
    avg = {}

    if 'affs' in available_preds:
        avg['affs'] = {}
        avg['mws'] = {}
    if 'lsds' in available_preds:
        avg['lsds'] = {}

    to_average = ['rand_split', 'rand_merge', 'voi_split', 'voi_merge', 'nvi_split', 'nvi_merge', 'nid', 'voi_sum', 'nvi_sum']

    for key in avg:
        for metric in to_average:

            if key == "mws":
                vals = [run[key][metric] for run in results['results']]
            else:
                vals = [run[key]['best'][metric] for run in results['results']]

            assert len(vals) == n_runs

            avg[key][metric] = [np.mean(vals),np.std(vals)]

    results['avg'] = avg

    #add meta
    results['raw_container'] = raw_container
    results['roi'] = roi
    results['downsampling_factor'] = factor
    results['denoise_wt'] = denoise_wt

    #dump
    with open(os.path.join(setup_path,f"results_{volume}.json"),"w") as f:
        json.dump(results,f,indent=4)
