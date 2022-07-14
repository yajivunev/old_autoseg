import sys
import json
import numpy as np
import daisy
import logging
import time
import os
import zarr
import waterz

from lsd import agglomerate_in_block
from evaluate_thresholds import evaluate
from watershed import watershed_in_block
from find_segments import get_connected_components
from funlib.segment.arrays import replace_values

import multiprocessing as mp

"""Script to perform the entire post-processing pipeline not blockwise."""


def write_segmentation(
        fragments_file,
        fragments_dataset,
        edges_collection,
        threshold,
        write_roi,
        voxel_size,
        run_type):
      
    fragments = daisy.open_ds(fragments_file,fragments_dataset)

    segmentation = daisy.prepare_ds(
        fragments_file,
        "segmentation_"+str(threshold),
        write_roi,
        voxel_size=voxel_size,
        dtype=np.uint64,
        write_roi=write_roi)

    lut_filename = f'seg_{edges_collection}_{int(threshold*100)}'

    lut_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    if run_type:
        lut_dir = os.path.join(lut_dir, run_type)
        logging.info(f"Run type set, using luts from {run_type} data")

    lut = os.path.join(
            lut_dir,
            lut_filename + '.npz')

    assert os.path.exists(lut), f"{lut} does not exist"

    logging.info("Reading fragment-segment LUT...")

    lut = np.load(lut)['fragment_segment_lut']

    logging.info(f"Found {len(lut[0])} fragments in LUT")

    num_segments = len(np.unique(lut[1]))
    logging.info(f"Relabelling fragments to {num_segments} segments")

    fragments = fragments.to_ndarray(write_roi)

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    relabelled = replace_values(fragments, lut[0], lut[1], out_array=relabelled)

    segmentation[write_roi] = relabelled


def pipeline(
        affs_file,
        affs_dataset,
        fragments_dataset,
        merge_function,
        gt_file=None,
        gt_dataset=None,
        roi_offset=None,
        roi_shape=None,
        fragments_in_xy=True,
        epsilon_agglomerate=0.0,
        filter_fragments=0.05,
        replace_sections=None,
        mask_file=None,
        mask_dataset=None,
        run_type=None):

    # open datasets

    affs = daisy.open_ds(affs_file,affs_dataset,mode='r')

    voxel_size = affs.voxel_size
    
    context = daisy.Coordinate([voxel_size[0]]*3)
    total_roi = affs.roi.grow(context, context)
    read_roi = total_roi
    write_roi = affs.roi

    fragments_file = affs_file
    block_directory = os.path.join(fragments_file, 'block_nodes')

    os.makedirs(block_directory, exist_ok=True)

    # prepare fragments dataset

    fragments = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        write_roi,
        voxel_size,
        np.uint64)

    num_voxels_in_block = (write_roi/affs.voxel_size).size

    # if mask is provided

    if mask_file is not None:

        logging.info("Reading mask from {}".format(mask_file))
        mask = daisy.open_ds(
            mask_file,
            mask_dataset,
            mode='r')

    else:

        mask = None

    # open RAG DB

    logging.info("Opening RAG file...")
    rag_provider = daisy.persistence.FileGraphProvider(
        directory=block_directory,
        chunk_size=write_roi.shape,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x']
        )

    # watershed the whole block

    block = daisy.Block(total_roi,read_roi,write_roi)

    logging.info("Performing watershed...")
    start = time.time()

    watershed_in_block(
        affs,
        block,
        context,
        rag_provider,
        fragments,
        num_voxels_in_block=num_voxels_in_block,
        mask=mask,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        filter_fragments=filter_fragments,
        replace_sections=replace_sections)
   
    logging.info(f"Time to watershed: {time.time() - start}")

    # agglomeration

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
    }[merge_function]

    edges_collection = 'edges_' + merge_function

    # reopening RAG file
    logging.info("Reopening RAG file...")
    rag_provider = daisy.persistence.FileGraphProvider(
        directory=block_directory,
        chunk_size=write_roi.shape,
        mode='r+',
        directed=False,
        edges_collection=edges_collection,
        position_attribute=['center_z', 'center_y', 'center_x']
        )

    logging.info("Performing agglomeration...")
    start = time.time()

    agglomerate_in_block(
            affs,
            fragments,
            rag_provider,
            block,
            merge_function=waterz_merge_function,
            threshold=1.0)

    # find segments

    logging.info(f"Agglomerated in {time.time() - start}")

    start = time.time()
    node_attrs = rag_provider.read_nodes(fragments.roi)
    edge_attrs = rag_provider.read_edges(fragments.roi,nodes=node_attrs)

    logging.info(f"Read graph in {time.time() - start}")

    if 'id' not in node_attrs:
        logging.info('No nodes found in roi %s' % roi)
        return

    nodes = node_attrs['id']
    edges = np.stack(
                [
                    edge_attrs['u'].astype(np.uint64),
                    edge_attrs['v'].astype(np.uint64)
                ],
            axis=1)

    scores = edge_attrs['merge_score'].astype(np.float32)

    logging.info(f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir = os.path.join(
        fragments_file,
        'luts',
        'fragment_segment')

    if run_type:
        out_dir = os.path.join(out_dir, run_type)

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [round(i,2) for i in np.arange(0,1,0.02)]

    start = time.time()

    # create LUTs
    
    for threshold in thresholds:

        # find segments
        
        logging.info(f"Creating LUT for threshold={threshold}")
        get_connected_components(
                nodes,
                edges,
                scores,
                threshold,
                edges_collection,
                out_dir)

    logging.info(f"Time to create LUTS: {time.time() - start}")
    # if gt_file exists, evaluate threshold

    if gt_file is not None:

        start = time.time()
        
        gt = daisy.open_ds(gt_file,gt_dataset,mode="r")

        if roi_offset:
            common_roi = daisy.Roi(roi_offset, roi_shape)

        else:
            common_roi = fragments.roi.intersect(gt.roi)
    
        logging.info(f"Opened gt dataset, roi={common_roi}")
        
        fragments = fragments[common_roi]
        gt = gt[common_roi]

        logging.info("Converting fragments to nd array...")
        fragments = fragments.to_ndarray()

        logging.info("Converting gt to nd array...")
        gt = gt.to_ndarray()

        manager = mp.Manager()
        metrics = manager.dict()

        metrics["voi"] = manager.dict()
        metrics["nvi"] = manager.dict()
        metrics["nid"] = manager.dict()

        pool = []

        for t in thresholds:

            p = mp.Process(target=evaluate, args=(t,fragments,gt,fragments_file,edges_collection,metrics,))
            pool.append(p)
            p.start()

        for p in pool: p.join()

        logging.info(f"Time to evaluate thresholds: {time.time() - start}")

    # get best thresholds if gt_file exists

    if gt_file is not None: 

        assert len(metrics["voi"]) != 0
        assert len(metrics["nvi"]) != 0
        assert len(metrics["nid"]) != 0

        logging.info("Best VOI,NVI,NID and respective thresholds: \n")
        best_voi = metrics["voi"][min(metrics["voi"].keys())]
        best_nvi = metrics["nvi"][min(metrics["nvi"].keys())]
        best_nid = metrics["nid"][min(metrics["nid"].keys())]

        voi_thresh,voi_split,voi_merge = best_voi['threshold'],best_voi['voi_split'],best_voi['voi_merge']
        nvi_thresh,nvi_split,nvi_merge = best_nvi['threshold'],best_nvi['nvi_split'],best_nvi['nvi_merge']
        nid,nid_thresh = best_nid['nid'],best_nid['threshold']

        logging.info("VOI: threshold= {} , VOI= {}, VOI_split= {} , VOI_merge= {}".format(voi_thresh,voi_split+voi_merge,voi_split,voi_merge))
        logging.info("NVI: threshold= {} , NVI= {}, NVI_split= {} , NVI_merge= {}".format(nvi_thresh,nvi_split+nvi_merge,nvi_split,nvi_merge))
        logging.info("NID: threshold= {} , NID= {}".format(nid_thresh,nid))

        thresholds = set([voi_thresh,nvi_thresh,nid_thresh])

    # else use default thresholds

    else:

        thresholds = [0.02,0.48,0.98]

    # write segmentation

    for threshold in thresholds:

        write_segmentation(
            fragments_file,
            fragments_dataset,
            edges_collection,
            threshold,
            write_roi,
            voxel_size,
            run_type)


if __name__ == "__main__":

    config = sys.argv[1]

    with open(config,"r") as f:
        config = json.load(f)

    results_file = os.path.join(config['affs_file'],'results.out')

    logging.basicConfig(
                filename=results_file,
                filemode='a',
                level=logging.INFO)

    pipeline(**config)
