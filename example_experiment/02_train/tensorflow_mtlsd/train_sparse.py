from __future__ import print_function
import json
import logging
import math
import numpy as np
import os
import sys
import tensorflow as tf

from gunpowder import *
from gunpowder.tensorflow import *
from lsd.gp import AddLocalShapeDescriptor

"""
Training script for sparsely labelled ground truth.
"""

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data'

#samples = [
#    'apical.zarr',
#    'spine.zarr'
#]

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

voxel_size = [50,2,2]

# needs to match order of samples (small to large)

def calc_max_padding(
        output_size,
        voxel_size,
        neighborhood=None,
        sigma=None,
        mode='shrink'):

    '''Calculate maximum labels padding needed.
    Args:
        output_size (array-like of ``int``):
            output size of network, in world units (a gunpowder coordinate)
        voxel_size (array-like of ``int``):
        
        voxel_size = [50,2,2]voxel size to use (a gunpowder coordinate)
        neighborhood (``list`` of array-like, optional):
            affinity neighborhood to use.
        sigma (``int``, optional):
            sigma if using lsds
        mode (``string``, optional):
            mode to use for snapping roi to grid, see gunpowder roi
            documentation for details
    Explanation:
        when padding labels, we need to ensure that each batch still contains at
        least 50% of GT data. Additionally, we need to also consider worst case
        45 degree rotation when elastically augmenting the data. Our max padding
        is calculated as follows:
            output_size = output size of network in world coordinates (i.e \
                    nanometers not voxels)
            method_padding = largest affinity neighborhood * voxel size (for \
                    affinities) or sigma * voxel size (for lsds)
            diagonal = diagonal between x and y dimensions (i.e square root \ of
            the sum of squares of x and y axes)
            max_padding = (output_size[z]/2, diagonal/2, diagonal/2) + \
                    method_padding
        we then need to ensure max padding is a multiple of the voxel size - use
        snap_to_grid for this (see gunpowder.roi.snap_to_grid())
    '''

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = Coordinate(
                            [np.abs(aff) for val in neighborhood \
                                    for aff in val if aff != 0]
                        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = Coordinate((sigma*3,)*3)

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = Roi(
                    (Coordinate(
                        [i/2 for i in [output_size[0], diag, diag]]) +
                        method_padding),
                    (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()

def train_until(max_iteration):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    unlabelled = ArrayKey('UNLABELLED')
    lsds = ArrayKey('PREDICTED_LSDS')
    gt_lsds = ArrayKey('GT_LSDS')
    gt_lsds_scale = ArrayKey('GT_LSDS_SCALE')
    lsds_gradient = ArrayKey('LSDS_GRADIENT')
    affs = ArrayKey('PREDICTED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    gt_affs_scale = ArrayKey('GT_AFFS_SCALE')
    gt_affs_mask = ArrayKey('GT_AFFS_MASK')
    affs_gradient = ArrayKey('AFFS_GRADIENT')

    input_shape = config['input_shape']
    output_shape = config['output_shape']

    voxel_size = Coordinate(voxel_size)
    input_size = Coordinate(input_shape)*voxel_size
    output_size = Coordinate(output_shape)*voxel_size

    #max labels padding calculated
    labels_padding = calc_max_padding(
                        output_size,
                        voxel_size,
                        sigma=100)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(gt_lsds_scale, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask,output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        lsds: request[gt_lsds],
        affs: request[gt_affs],
        affs_gradient: request[gt_affs],
        lsds_gradient: request[gt_lsds]
    })

    data_sources = tuple(
        ZarrSource(
            os.path.join(data_dir, sample),
            datasets = {
                raw: 'clahe_raw/s0',
                labels: 'labels_updated/s0',
                labels_mask: 'labels_mask/s0',
                unlabelled: 'unlabelled/s0'
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False),
                unlabelled: ArraySpec(interpolatable=False)
            }
        ) +
        Normalize(raw) +
        Pad(raw, None) +
        Pad(labels, labels_padding) +
        Pad(labels_mask, labels_padding) +
        Pad(unlabelled, labels_padding) +
        RandomLocation(min_masked=0.5, mask=unlabelled)
        for sample in samples
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[2, 50, 50],
            jitter_sigma=[0, 2, 2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8) +
        SimpleAugment(transpose_only = [1,2]) +
        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        GrowBoundary(labels, labels_mask, steps=1, only_xy=True) +
        AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            mask=gt_lsds_scale,
            sigma=100,
            downsample=2) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabelled,
            affinities_mask=gt_affs_mask) +
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask) +
        IntensityScaleShift(raw, 2,-1) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'train_net',
            optimizer=config['optimizer'],
            loss=config['loss'],
            inputs={
                config['raw']: raw,
                config['gt_lsds']: gt_lsds,
                config['loss_weights_lsds']: gt_lsds_scale,
                config['gt_affs']: gt_affs,
                config['loss_weights_affs']: gt_affs_scale,
            },
            outputs={
                config['lsds']: lsds,
                config['affs']: affs
            },
            gradients={
                config['affs']: affs_gradient,
                config['lsds']: lsds_gradient
            },
            summary=config['summary'],
            log_dir='log',
            save_every=1000) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_lsds: 'volumes/gt_lsds',
                lsds: 'volumes/pred_lsds',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                affs_gradient: 'volumes/affs_gradient',
                lsds_gradient: 'volumes/lsds_gradient'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=500,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":

    iteration = int(sys.argv[1])
    train_until(iteration)
