import json
import logging
import math
import numpy as np
import os
import sys
import torch

from model import MtlsdModel, WeightedMSELoss, calc_max_padding

from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from lsd.gp import AddLocalShapeDescriptor

# example training script for mtlsd model for neuron segmentation

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../01_data'

# uncomment and list training zarr directories below

#samples = [
#    'sample_A.zarr',
#]

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

batch_size = 1

def train(
        max_iteration,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        input_shape,
        voxel_size,
        sigma,
        **kwargs):

    model = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4,
            betas=(0.95,0.999),
            eps=1e-08)

    if 'output_shape' not in kwargs:
        output_shape = model.forward(torch.empty(size=[1,1]+input_shape))[0].shape[2:]
        with open("config.json","r") as f:
            config = json.load(f)
            
        config['output_shape'] = list(output_shape)
            
        with open("config.json","w") as f:
            json.dump(config,f)

    else: output_shape = kwargs.get("output_shape")

    output_shape = Coordinate(tuple(output_shape))
    input_shape = Coordinate(tuple(input_shape))

    print("output shape: ",tuple(output_shape))

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    pred_affs = ArrayKey('PRED_AFFS')
    gt_affs = ArrayKey('GT_AFFS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')
    pred_lsds = ArrayKey('PRED_LSDS')
    gt_lsds = ArrayKey('GT_LSDS')
    lsds_weights = ArrayKey('LSDS_WEIGHTS')

    voxel_size = Coordinate(tuple(voxel_size))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
                        output_size,
                        voxel_size,
                        sigma=sigma)

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    data_sources = tuple(
            ZarrSource(
                    os.path.join(data_dir, sample),
                    {
                        raw: 'volumes/raw',
                        labels: 'volumes/labels/neuron_ids',
                        labels_mask: 'volumes/labels/mask'
                    },
                    {
                        raw: ArraySpec(interpolatable=True),
                        labels: ArraySpec(interpolatable=False),
                        labels_mask: ArraySpec(interpolatable=False)
                    }
                ) +
            Normalize(raw) +
            Pad(raw, None) +
            Pad(labels, labels_padding) +
            Pad(labels_mask, labels_padding) +
            RandomLocation(min_masked=0.5,mask=labels_mask)
            for sample in samples
        )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=[4,40,40],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8)

    train_pipeline += SimpleAugment(transpose_only=[1, 2])
    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    train_pipeline += GrowBoundary(
            labels,
            mask=labels_mask,
            steps=1,
            only_xy=True)

    train_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            mask=lsds_weights,
            sigma=sigma,
            downsample=2)

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs)

    train_pipeline += BalanceLabels(
            gt_affs,
            affs_weights)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(batch_size)

    train_pipeline += PreCache(
            cache_size=40,
            num_workers=10)

    train_pipeline += Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            loss_inputs={
                'lsds_prediction': pred_lsds,
                'lsds_target': gt_lsds,
                'lsds_weights': lsds_weights,
                'affs_prediction': pred_affs,
                'affs_target': gt_affs,
                'affs_weights': affs_weights
            },
            outputs={
                0: pred_lsds,
                1: pred_affs
            },
            save_every=1000,
            log_dir='log')

    train_pipeline += Squeeze([raw])
    train_pipeline += Squeeze([raw, gt_affs, pred_affs, gt_lsds, pred_lsds])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                labels_mask: 'labels_mask',
                gt_affs: 'gt_affs',
                gt_lsds: 'gt_lsds',
                pred_affs: 'pred_affs',
                pred_lsds: 'pred_lsds'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=500,
            output_filename='batch_{iteration}.zarr')

    train_pipeline += PrintProfilingStats(every=10)

    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    #config_file = sys.argv[1]

    with open('config.json','r') as f:
        config = json.load(f)

    train(**config)
