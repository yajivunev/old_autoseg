import json
import gunpowder as gp
import logging
import math
import numpy as np
import os

import torch
from model import UnetModel,WeightedBCELoss

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

data_dir = '../../01_data/'

samples = ['fhltd.zarr','fpnct.zarr','fwngv.zarr','xrzct.zarr']


def get_sections(data_dir, samples):

    sections = []

    for sample in samples:
        csv_path = os.path.join(data_dir, sample, 'csvs')

        non_empty_sections = [int(x.split('_')[-1].split('.')[0]) for x in os.listdir(csv_path)]

        sections.append(non_empty_sections)

    # sections = [1,]*4

    return {samp:sec for samp,sec in zip(samples, sections)}


sections = get_sections(data_dir, samples)


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

    loss = WeightedBCELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)

    model.apply(init_weights)

    if 'output_shape' not in kwargs:

        output_shape = model.forward(torch.empty(size=[1,1]+input_shape)).shape[2:]
        config['output_shape'] = output_shape

        with open("config.json","w") as f:
            json.dump(config,f)
    else:

        output_shape = config['output_shape']
	
    print(f"output_shape:{output_shape}")
    input_shape = gp.Coordinate(tuple(input_shape))
    output_shape = gp.Coordinate(output_shape)
    labels_padding = output_shape

    voxel_size = gp.Coordinate(tuple(voxel_size))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    points = gp.GraphKey('POINTS')
    raster = gp.ArrayKey('RASTER')
    pred_raster = gp.ArrayKey('PRED_RASTER')
    unlabelled = gp.ArrayKey('UNLABELLED')
    raster_weights=gp.ArrayKey('RASTER_WEIGHTS')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(points, output_size)
    request.add(raster, output_size)
    request.add(pred_raster, output_size)
    request.add(raster_weights, output_size)
    request.add(unlabelled, output_size)

    sources = (tuple((
            gp.ZarrSource(
                filename=os.path.join(data_dir,sample),
                datasets={
                    raw: f'2d_raw/{i}',
                    unlabelled: f'2d_unlabelled/{i}'},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True),
                    unlabelled: gp.ArraySpec(interpolatable=True)}) +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Pad(unlabelled, None),

            gp.CsvPointsSource(
                filename=os.path.join(data_dir,sample,f'csvs/section_{i}.csv'),
                points=points,
                ndims=2,
                scale=[2,2]) +
            gp.Pad(points, labels_padding) +
            gp.RasterizeGraph(
                points,
                raster,
                array_spec=gp.ArraySpec(voxel_size=voxel_size,dtype=np.uint8),
                settings=gp.RasterizationSettings(
                    radius=(10,10),
                    mode='ball')
                )
            ) + gp.MergeProvider() + gp.RandomLocation(ensure_nonempty=points) for i in sec)
            for sample,sec in sections.items())


    sources = tuple(y for x in sources for y in x)
 
    pipeline = sources

    pipeline += gp.RandomProvider()

    #pipeline += gp.SimpleAugment()

    #pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)
   
    pipeline += gp.BalanceLabels(raster,raster_weights,mask=unlabelled)

    pipeline += gp.Normalize(raster,factor=1.0)
    
    pipeline += gp.Unsqueeze([raw,raster,raster_weights])

    pipeline += gp.Stack(batch_size)

    pipeline += gp.IntensityScaleShift(raw,2,-1)

    pipeline += gp.PreCache(
            cache_size=100,
            num_workers=64)

    pipeline += gp.torch.Train(
            model,
            loss=loss,
            optimizer=optimizer,
            inputs={
                'input': raw
            },
            outputs={
                0: pred_raster
            },
            loss_inputs={
                'pred': pred_raster,
                'target': raster,
                'weights': raster_weights,
            },
            array_specs={
                pred_raster: gp.ArraySpec(interpolatable=True,voxel_size=voxel_size)
            },
            save_every=1000,
            log_dir='log')

    pipeline += gp.Squeeze([raw,raster,raster_weights,pred_raster],axis=1)

    pipeline += gp.IntensityScaleShift(raw,0.5,0.5)

    pipeline += gp.Snapshot({
            raw: 'raw',
            raster: 'raster',
            pred_raster: 'pred',
            raster_weights: 'weights'
        },
        every=1000,
        output_filename='batch_{iteration}.zarr',
        dataset_dtypes={
                raster: np.float32
            }
        )

    with gp.build(pipeline):
        for i in range(max_iteration):
            pipeline.request_batch(request)

if __name__ == '__main__':

    with open('config.json','r') as f:
        config = json.load(f)

    train(**config)
