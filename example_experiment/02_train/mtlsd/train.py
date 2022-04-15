import os
import json
import logging
import math
import numpy as np
import torch
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import MtLsdModule,calc_max_padding

from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
from lsd.gp import AddLocalShapeDescriptor

logging.basicConfig(level=logging.INFO)

data_dir = '../../01_data'
samples = ['cremi_sample_a.zarr','cremi_sample_b.zarr','cremi_sample_c.zarr']

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def init_weights(m):
    if isinstance(m, (torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        m.bias.data.fill_(0.01)
        
        
class pipeline_build(object):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        try:
            self.pipeline.setup()
        except:
            logger.error("something went wrong during the setup of the pipeline, calling tear down")
            self.pipeline.internal_teardown()
            logger.debug("tear down completed")
            raise
        return self.pipeline

    def __exit__(self, type, value, traceback):
        #logger.debug("leaving context, tearing down pipeline")
        #self.pipeline.internal_teardown()
        #logger.debug("tear down completed")
        pass

    
class Dataset(torch.utils.data.Dataset):
  
    def __init__(self):
        
        #with pipeline_build(pipeline) as b:
        self.num_batches = max_iteration
        
    def __len__(self):
        
        return self.num_batches
    
    def __getitem__(self,index):
        
        batch = pipeline.request_batch(request)
        
        array_keys = list((batch.arrays).keys())
        
        arrays = tuple([batch[key].data for key in array_keys if key not in [ArrayKey('GT_LABELS')]])

        return arrays


def request_and_pipeline(
        input_shape,
        output_shape,
        voxel_size,
        sigma,
        downsample):

    output_shape = Coordinate(tuple(output_shape))
    input_shape = Coordinate(tuple(input_shape))

    raw_fr = ArrayKey('RAW_FR')
    labels_fr = ArrayKey('GT_LABELS_FR')
    labels_mask_fr = ArrayKey('GT_LABELS_MASK_FR')

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    gt_lsds = ArrayKey('GT_LSDS')
    lsds_weights = ArrayKey('LSDS_WEIGHTS')
    gt_affs = ArrayKey('GT_AFFS')
    affs_weights = ArrayKey('AFFS_WEIGHTS')

    downsampling = Coordinate((1,downsample,downsample))
    voxel_size = Coordinate(tuple(voxel_size)) * downsampling
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
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)

    data_sources = tuple(
            ZarrSource(
                    os.path.join(data_dir, sample),
                    {
                        raw_fr: 'raw',
                        labels_fr: 'labels',
                        labels_mask_fr: 'labels_mask'
                    },
                    {
                        raw_fr: ArraySpec(interpolatable=True),
                        labels_fr: ArraySpec(interpolatable=False),
                        labels_mask_fr: ArraySpec(interpolatable=False)
                    }
                ) +
            Normalize(raw_fr) +
            Pad(raw_fr, None) +
            Pad(labels_fr, labels_padding) +
            Pad(labels_mask_fr, labels_padding) +
            RandomLocation(min_masked=0.5,mask=labels_mask_fr) +
            DownSample(raw_fr, (1, downsample, downsample), raw) +
            DownSample(labels_fr, (1, downsample, downsample), labels) +
            DownSample(labels_mask_fr, (1, downsample, downsample), labels_mask)
            for sample in samples
        )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=[4,int(40/downsample),int(40/downsample)],
            jitter_sigma=[0,2,2],
            rotation_interval=[0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=int(28/downsample),
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
            labels_mask=labels_mask,
            mask=lsds_weights,
            sigma=sigma,
            downsample=2)

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs)

    train_pipeline += BalanceLabels(
            gt_affs,
            affs_weights,
            mask=labels_mask)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(1)

    train_pipeline += PreCache(32,32)

    train_pipeline += Squeeze([raw,gt_affs,affs_weights,gt_lsds,lsds_weights])

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                labels_mask: 'labels_mask',
                gt_affs: 'gt_affs',
                gt_lsds: 'gt_lsds',
                affs_weights: 'affs_weights',
                lsds_weights: 'lsds_weights'
            },
            dataset_dtypes={
                labels: np.uint64,
                gt_affs: np.float32
            },
            every=500,
            output_filename='batch_{id}.zarr')

    train_pipeline += PrintProfilingStats(every=100)

    return request,train_pipeline


if __name__ == "__main__":
    
    with open('config.json','r') as f:
        config = json.load(f)
    
    max_iteration = config['max_iteration']
    in_channels = config['in_channels']
    num_fmaps = config['num_fmaps']
    fmap_inc_factor = config['fmap_inc_factor']
    downsample_factors = config['downsample_factors']
    kernel_size_down = config['kernel_size_down']
    kernel_size_up = config['kernel_size_up']
    input_shape = config['input_shape']
    output_shape = config['output_shape']
    voxel_size = config['voxel_size']
    sigma = config['sigma']
    downsample = config['downsample']
   
    batch_size = 1
    num_workers = 0

    request,pipeline = request_and_pipeline(
        input_shape,
        output_shape,
        voxel_size,
        sigma,
        downsample)
   
    with pipeline_build(pipeline) as b:
        pipeline = b

    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
   
    logger = pl.loggers.TensorBoardLogger(".",name="log",log_graph=True)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',save_top_k=-1,save_on_train_epoch_end=True,every_n_train_steps=500)

    model = MtLsdModule(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    model.train()

    trainer = pl.Trainer(
            max_epochs=-1,
            accelerator="gpu",
            devices=4,
            precision=16,
            amp_backend='native',
            benchmark=True,
            logger=logger,
            log_every_n_steps=100,
            callbacks=[checkpoint_callback],
            strategy=pl.strategies.BaguaStrategy(algorithm="async"))
            #strategy=pl.strategies.DDPStrategy(
            #    gradient_as_bucket_view=True,
            #    static_graph=True,
            #    find_unused_parameters=False))

    if ('last_checkpoint' in config.keys() and config['last_checkpoint'] != ""):
            print(f'Loading from checkpoint:{config["last_checkpoint"]}')
            trainer.fit(model,dataloader,ckpt_path=config['last_checkpoint'])

    else:
        print("No checkpoint found, initializing weights...")
        model.apply(init_weights)
        trainer.fit(model,dataloader)

    config['last_checkpoint']=checkpoint_callback.best_model_path

    with open('config.json','w') as f:
        json.dump(config,f)
