import daisy
import numpy as np
import os
import sys
import waterz
import zarr
import logging

from pipeline.watershed import watershed_from_affinities

def get_segmentation(affs, threshold, labels_mask=None):

    fragments = watershed_from_affinities(
            affs,
            labels_mask)[0]
    
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation

if __name__ == "__main__":

    affs_file = sys.argv[1]
    affs_dataset = sys.argv[2]
    threshold = float(sys.argv[3])

    affs = daisy.open_ds(affs_file,affs_dataset)
    roi = affs.roi
    voxel_size = affs.voxel_size
  
    logging.info(f"affs_file: {affs_file}, dataset: {affs_dataset}, roi: {roi}, voxel_size: {voxel_size}")

    affs = affs.to_ndarray()
    
    segmentation = get_segmentation(affs,threshold)

    seg_ds = daisy.prepare_ds(affs_file,f"segmentation_{threshold}",roi,voxel_size,dtype=np.uint64)

    logging.info(f"writing to {affs_file}, dataset: segmentation_{threshold}")

    seg_ds[roi] = segmentation
