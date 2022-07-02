import glob
import json
import logging
import math
import numpy as np
import os
import sys
import torch

from utils import SwapAxes, BinaryDilation, BumpBackground, UnbumpBackground

from model import UnetModel, WeightedLoss
import gunpowder as gp
from lsd.gp import AddLocalShapeDescriptor

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../../../data'

samples = ['xrzct.zarr']


def get_sections(data_dir, samples):

    sections = []

    for sample in samples:
        csv_path = os.path.join(data_dir, sample, 'csvs')

        non_empty_sections = [int(x.split('_')[-1].split('.')[0]) for x in os.listdir(csv_path)]

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

        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape)).shape[2:]
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
    raster = gp.ArrayKey('RASTER')
    unlabelled_mask = gp.ArrayKey('UNLABELLED_MASK')
    gt_lsd = gp.ArrayKey('GT_LSD')
    pred_lsd = gp.ArrayKey('PRED_LSD')
    raster = gp.ArrayKey('GT_MASK')
    pred_mask = gp.ArrayKey('PRED_MASK')
    weights = gp.ArrayKey('WEIGHTS')
    
    labels_padding = output_shape

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(raster, output_size)
    request.add(unlabelled_mask, output_size)
    request.add(gt_lsd, output_size)
    request.add(pred_lsd, output_size)
    request.add(weights, output_size)
    request.add(raster, output_size)
    request.add(pred_mask, output_size)

    sources = (tuple(
            gp.ZarrSource(
                filename=os.path.join(data_dir,sample),
                datasets={
                    raw: f'2d_raw/{i}',
                    raster: f'2d_raster/{i}',
                    unlabelled_mask: f'2d_unlabelled/{i}'},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    raster: gp.ArraySpec(interpolatable=True),
                    unlabelled_mask: gp.ArraySpec(interpolatable=True)}) +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Pad(raster, labels_padding) +
            gp.Pad(unlabelled_mask, labels_padding) +
            #gp.RandomLocation() +
            gp.RandomLocation(mask=unlabelled_mask, min_masked=0.05) for i in sec[100:120])
            #gp.Reject(mask=unlabelled_mask, min_masked=0.05) for i in sec)
            for sample,sec in sections.items())


    sources = tuple(y for x in sources for y in x)

    train_pipeline = sources

    train_pipeline += gp.RandomProvider()

    train_pipeline += BinaryDilation(raster,iterations=2)

    train_pipeline += BumpBackground(raster)

    train_pipeline += AddLocalShapeDescriptor(
            raster,
            gt_lsd,
            unlabelled=unlabelled_mask,
            sigma=4,
            downsample=1,
            component=None)

    train_pipeline += UnbumpBackground(raster)
 
    train_pipeline += gp.BalanceLabels(
            raster,
            weights,
            unlabelled_mask)

    train_pipeline += gp.IntensityScaleShift(raw, 2,-1)

    train_pipeline += gp.Unsqueeze([raw, weights, raster]) #c,y,x
    train_pipeline += gp.Stack(batch_size) #n,c,y,x

    #train_pipeline += gp.PreCache(
    #        cache_size=16,
    #        num_workers=64)

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
                2: weights,
                3: pred_mask,
                4: raster,
                5: weights
            },
            outputs={
                0: pred_lsd,
                1: pred_mask
            },
            save_every=1000,
            log_dir='log')

    train_pipeline += SwapAxes([raw,weights,gt_lsd,pred_lsd,raster,pred_mask],(0,1)) #c,n,y,x

    train_pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += gp.Snapshot({
                raw: 'raw',
                raster: 'raster',
                unlabelled_mask: 'unlabelled',
                gt_lsd: 'gt_lsd',
                pred_lsd: 'pred_lsd',
                weights: 'weights',
                raster: 'raster',
                pred_mask: 'pred_mask',
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
