import json
import logging
import math
import numpy as np
import os
import sys
import torch

from model import MtlsdModel, WeightedMSELoss
from utils import calc_max_padding, BumpBackground, UnbumpBackground

import gunpowder as gp
from lsd.gp import AddLocalShapeDescriptor

# example training script for mtlsd model for neuron segmentation

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../../../../data'

# uncomment and list training zarr directories below

samples = ['voljo_3.zarr']

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def init_weights(m):
    if isinstance(m, (torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        #m.bias.data.fill_(0.01)

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
        downsample,
        batch_size,
        **kwargs):

    model = MtlsdModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    model.apply(init_weights)

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1e-4,
            betas=(0.95,0.999))

    if 'output_shape' not in kwargs:
        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape))[0].shape[2:]
        with open("config.json","r") as f:
            config = json.load(f)
            
        config['output_shape'] = list(output_shape)
            
        with open("config.json","w") as f:
            json.dump(config,f)

    else: output_shape = kwargs.get("output_shape")

    output_shape = gp.Coordinate(tuple(output_shape))
    input_shape = gp.Coordinate(tuple(input_shape))

    raw_fr = gp.ArrayKey('RAW_FR')
    labels_fr = gp.ArrayKey('GT_LABELS_FR')
    labels_mask_fr = gp.ArrayKey('GT_LABELS_MASK_FR')
    unlabelled_fr = gp.ArrayKey('UNLABELLED_FR')

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('GT_LABELS')
    labels_mask = gp.ArrayKey('GT_LABELS_MASK')
    unlabelled = gp.ArrayKey('UNLABELLED')
    
    pred_affs = gp.ArrayKey('PRED_AFFS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    affs_mask = gp.ArrayKey('AFFS_MASK')
    
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    gt_lsds = gp.ArrayKey('GT_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')

    downsampling = gp.Coordinate((1,downsample,downsample))
    voxel_size = gp.Coordinate(tuple(voxel_size)) * downsampling
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    labels_padding = calc_max_padding(
                        output_size,
                        voxel_size,
                        sigma=sigma)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_mask, output_size)

    data_sources = tuple(
            gp.ZarrSource(
                    os.path.join(data_dir, sample),
                    {
                        raw_fr: 'raw',
                        labels_fr: 'neuron',
                        labels_mask_fr: 'labels_mask',
                        unlabelled_fr: 'unlabelled',
                    },
                    {
                        raw_fr: gp.ArraySpec(interpolatable=True),
                        labels_fr: gp.ArraySpec(interpolatable=False),
                        labels_mask_fr: gp.ArraySpec(interpolatable=False),
                        unlabelled_fr: gp.ArraySpec(interpolatable=False),
                    }
                ) +
            gp.Normalize(raw_fr) +
            gp.Pad(raw_fr, None) +
            gp.Pad(labels_fr, labels_padding) +
            gp.Pad(labels_mask_fr, labels_padding) +
            gp.Pad(unlabelled_fr, labels_padding) +
            gp.RandomLocation(min_masked=0.5,mask=unlabelled_fr) +
            gp.DownSample(raw_fr, (1, downsample, downsample), raw) +
            gp.DownSample(labels_fr, (1, downsample, downsample), labels) +
            gp.DownSample(labels_mask_fr, (1, downsample, downsample), labels_mask) +
            gp.DownSample(unlabelled_fr, (1, downsample, downsample), unlabelled)
            for sample in samples
        )

    train_pipeline = data_sources

    train_pipeline += gp.RandomProvider()

    train_pipeline += gp.ElasticAugment(
            control_point_spacing=[2,int(50/downsample),int(50/downsample)],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=int(28/downsample),
            subsample=8)

    train_pipeline += gp.SimpleAugment(transpose_only=[1, 2])
    train_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

    train_pipeline += gp.GrowBoundary(
            labels,
            mask=labels_mask,
            steps=1,
            only_xy=True)

    train_pipeline += gp.GrowBoundary(
            labels_mask,
            steps=5,
            only_xy=True)

    train_pipeline += BumpBackground(labels)

    train_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            mask=lsds_weights,
            labels_mask=labels_mask,
            sigma=sigma,
            downsample=1)

    train_pipeline += UnbumpBackground(labels)

    train_pipeline += gp.AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabelled,
            affinities_mask=affs_mask)

    train_pipeline += gp.BalanceLabels(
            gt_affs,
            affs_weights,
            mask=affs_mask)

    train_pipeline += gp.IntensityScaleShift(raw, 2,-1)

    train_pipeline += gp.Unsqueeze([raw]) #1,z,y,x
    train_pipeline += gp.Stack(batch_size) #n,c,z,y,x

    train_pipeline += gp.PreCache(
            cache_size=40,
            num_workers=16)

    train_pipeline += gp.torch.Train(
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
            save_every=2500,
            log_dir='log')

    train_pipeline += gp.Squeeze([raw,gt_affs,pred_affs,gt_lsds,pred_lsds,labels,affs_weights,lsds_weights]) #c,z,y,x

    train_pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += gp.Snapshot({
                raw: 'raw',
                labels: 'labels',
                labels_mask: 'labels_mask',
                gt_affs: 'gt_affs',
                gt_lsds: 'gt_lsds',
                pred_affs: 'pred_affs',
                pred_lsds: 'pred_lsds',
                lsds_weights: 'lsds_weights',
                affs_weights: 'affs_weights'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=25,
            output_filename='batch_{iteration}.zarr')

    train_pipeline += gp.PrintProfilingStats(every=500)

    with gp.build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    with open('config.json','r') as f:
        config = json.load(f)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train(**config)
