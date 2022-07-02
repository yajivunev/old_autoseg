import glob
import json
import logging
import math
import numpy as np
import os
import sys
import torch

from utils import SwapAxes, BinaryDilation, BoostWeights

from model import UnetModel, WeightedLoss
import gunpowder as gp
from lsd.gp import AddLocalShapeDescriptor

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../01_data'

samples = ['xrzct.zarr','fhltd.zarr','fpnct.zarr','fwngv.zarr']


def get_sections(data_dir, samples):

    sections = []

    for sample in samples:
        ds_path = os.path.join(data_dir, sample, '2d_traces_mask')

        non_empty_sections = [int(x) for x in os.listdir(ds_path) if not x.startswith(".z")]

        sections.append(non_empty_sections)

    # sections = [1,]*4

    return {samp:sec for samp,sec in zip(samples, sections)}

sections = get_sections(data_dir,samples)


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
        batch_size,
        **kwargs):


    model = UnetModel(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    loss = WeightedLoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            betas=(0.95,0.999))

    model.apply(init_weights)

    if 'output_shape' not in kwargs:

        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape))[0].shape[2:]
        config['output_shape'] = output_shape

        with open("config.json","w") as f:
            json.dump(config,f)
    else:

        output_shape = config['output_shape']

    print(f"output_shape:{output_shape}")
    input_shape = gp.Coordinate(tuple(input_shape))
    output_shape = gp.Coordinate(output_shape)

    voxel_size = gp.Coordinate(tuple(voxel_size))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    gt_mask = gp.ArrayKey('GT_MASK')
    traces_mask = gp.ArrayKey('TRACES_MASK')
    mito_mask = gp.ArrayKey('MITO_MASK')
    unlabelled_mask = gp.ArrayKey('UNLABELLED_MASK')
    gt_lsd = gp.ArrayKey('GT_LSD')
    pred_lsd = gp.ArrayKey('PRED_LSD')
    lsd_weights = gp.ArrayKey('LSD_WEIGHTS')
    gt_affs = gp.ArrayKey('GT_AFFS')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')
    affs_mask = gp.ArrayKey('AFFS_MASK')
    
    labels_padding = output_shape

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_mask, output_size)
    request.add(traces_mask, output_size)
    request.add(mito_mask, output_size)
    request.add(unlabelled_mask, output_size)
    request.add(gt_lsd, output_size)
    request.add(pred_lsd, output_size)
    request.add(lsd_weights, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(affs_mask, output_size)
    request.add(gt_affs, output_size)

    sources = (tuple(
            gp.ZarrSource(
                filename=os.path.join(data_dir,sample),
                datasets={
                    raw: f'2d_raw/{i}',
                    gt_mask: f'2d_combined_vesicles/{i}',
                    traces_mask: f'2d_traces_mask/{i}',
                    mito_mask: f'2d_mito/{i}',
                    unlabelled_mask: f'2d_unlabelled/{i}'},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    gt_mask: gp.ArraySpec(interpolatable=False),
                    traces_mask: gp.ArraySpec(interpolatable=False),
                    mito_mask: gp.ArraySpec(interpolatable=False),
                    unlabelled_mask: gp.ArraySpec(interpolatable=False)}) +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Pad(gt_mask, labels_padding) +
            gp.Pad(traces_mask, labels_padding) +
            gp.Pad(mito_mask, labels_padding) +
            gp.Pad(unlabelled_mask, labels_padding) +
            #gp.RandomLocation() +
            gp.RandomLocation(mask=traces_mask, min_masked=0.0005) for i in sec)
            #gp.Reject(mask=unlabelled_mask, min_masked=0.05) for i in sec)
            for sample,sec in sections.items())


    sources = tuple(y for x in sources for y in x)

    train_pipeline = sources

    train_pipeline += gp.RandomProvider()

    train_pipeline += BinaryDilation(traces_mask,iterations=2)

    train_pipeline += AddLocalShapeDescriptor(
            gt_mask,
            gt_lsd,
            mask=lsd_weights,
            sigma=4,
            downsample=1,
            component=None)
 
    train_pipeline += gp.AddAffinities(
            affinity_neighborhood=[
                [-1, 0],
                [0, -1]],
            labels=gt_mask,
            affinities=gt_affs,
            labels_mask=unlabelled_mask,
            affinities_mask=affs_mask)

    train_pipeline += gp.BalanceLabels(
            gt_affs,
            affs_weights,
            affs_mask)

    train_pipeline += BoostWeights(
            traces_mask,
            affs_weights,
            10)

    train_pipeline += BoostWeights(
            mito_mask,
            affs_weights,
            10)

    train_pipeline += BoostWeights(
            traces_mask,
            lsd_weights,
            5)

    train_pipeline += BoostWeights(
            mito_mask,
            lsd_weights,
            5)

    train_pipeline += gp.IntensityScaleShift(raw, 2,-1)

    train_pipeline += gp.Unsqueeze([raw]) #c,y,x
    train_pipeline += gp.Stack(batch_size) #n,c,y,x

    train_pipeline += gp.PreCache(
            cache_size=16,
            num_workers=64)

    train_pipeline += gp.torch.Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            loss_inputs={
                0: pred_lsd,
                1: gt_lsd,
                2: lsd_weights,
                3: pred_affs,
                4: gt_affs,
                5: affs_weights
            },
            outputs={
                0: pred_lsd,
                1: pred_affs
            },
            save_every=1000,
            log_dir='log')

    train_pipeline += SwapAxes([raw,lsd_weights,affs_weights,gt_lsd,pred_lsd,gt_affs,pred_affs],(0,1)) #c,n,y,x

    train_pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += gp.Snapshot({
                raw: 'raw',
                gt_mask: 'gt_mask',
                traces_mask: 'traces_mask',
                mito_mask: 'mito_mask',
                unlabelled_mask: 'unlabelled',
                gt_lsd: 'gt_lsd',
                pred_lsd: 'pred_lsd',
                lsd_weights: 'lsd_weights',
                pred_affs: 'pred_affs',
                affs_weights: 'affs_weights',
                gt_affs: 'gt_affs',
            },
            every=1000,
            output_filename='batch_{iteration}.zarr',
            )

    train_pipeline += gp.PrintProfilingStats(every=500)

    with gp.build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

if __name__ == '__main__':

    with open('config.json','r') as f:
        config = json.load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train(**config)
