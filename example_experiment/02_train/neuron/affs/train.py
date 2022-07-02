import os
import json
import logging
import math
import numpy as np
import torch
import time
import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import UNetModule,calc_max_padding

from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *

#logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

samples = [] #insert samples here

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
        pass

    
class Dataset(torch.utils.data.Dataset):
  
    def __init__(self,pipeline,request):
        
        self.num_batches = max_iteration
        self.pipeline = pipeline
        self.request = request
        
    def __len__(self):
        
        return self.num_batches
    
    def __getitem__(self,index):
        
        batch = self.pipeline.request_batch(self.request)
        
        array_keys = list((batch.arrays).keys())
        
        arrays = [batch[key].data for key in array_keys if key not in [ArrayKey('GT_LABELS'),ArrayKey('GT_LABELS_MASK'),ArrayKey('UNLABELLED')]]

        arrays.append(batch.id)

        return tuple(arrays)


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
    unlabelled_fr = ArrayKey('UNLABELLED_FR')

    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_mask = ArrayKey('GT_LABELS_MASK')
    unlabelled = ArrayKey('UNLABELLED')
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
    request.add(unlabelled,output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)

    data_sources = tuple(
            ZarrSource(
                    sample,
                    {
                        raw_fr: 'clahe_raw',
                        labels_fr: 'labels',
                        labels_mask_fr: 'labels_mask',
                        unlabelled_fr: 'unlabelled'
                    },
                    {
                        raw_fr: ArraySpec(interpolatable=True),
                        labels_fr: ArraySpec(interpolatable=False),
                        labels_mask_fr: ArraySpec(interpolatable=False),
                        unlabelled_fr: ArraySpec(interpolatable=False)
                    }
                ) +
            Normalize(raw_fr) +
            Pad(raw_fr, None) +
            Pad(labels_fr, labels_padding) +
            Pad(labels_mask_fr, labels_padding) +
            Pad(unlabelled_fr, labels_padding) +
            RandomLocation(mask=unlabelled_fr,min_masked=0.5) +
            DownSample(raw_fr, (1, downsample, downsample), raw) +
            DownSample(labels_fr, (1, downsample, downsample), labels) +
            DownSample(labels_mask_fr, (1, downsample, downsample), labels_mask) +
            DownSample(unlabelled_fr, (1, downsample, downsample), unlabelled)
            for sample in samples
        )

    train_pipeline = data_sources

    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            control_point_spacing=[8,int(25/downsample),int(25/downsample)],
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

    train_pipeline += AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            unlabelled=unlabelled)

    train_pipeline += BalanceLabels(
            gt_affs,
            affs_weights,
            mask=unlabelled)

    train_pipeline += IntensityScaleShift(raw, 2,-1)

    train_pipeline += Unsqueeze([raw])
    train_pipeline += Stack(1)

    train_pipeline += Squeeze([raw,gt_affs,affs_weights])

    train_pipeline += PreCache(10,8)

    train_pipeline += Snapshot({
                raw: 'raw',
                labels: 'labels',
                labels_mask: 'labels_mask',
                gt_affs: 'gt_affs',
                affs_weights: 'affs_weights',
            },
            dataset_dtypes={
                labels: np.uint8,
                gt_affs: np.float32
            },
            every=1000,
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
    voxel_size = config['voxel_size']
    sigma = config['sigma']
    downsample = config['downsample']
    batch_size = config['batch_size']

    model = UNetModule(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            batch_size,
            input_shape)

    model.train()


    if 'output_shape' not in config:

        output_shape = model.forward(model.example_input_array)[0].shape[2:]
        config['output_shape'] = output_shape

        with open("config.json","w") as f:
            json.dump(config,f)
    else:

        output_shape = config['output_shape']


    request,pipeline = request_and_pipeline(
        input_shape,
        output_shape,
        voxel_size,
        sigma,
        downsample)
   
    with pipeline_build(pipeline) as b:
        pipeline = b

    dataset = Dataset(pipeline,request)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=seed_worker)
   
    logger = pl.loggers.TensorBoardLogger(".",name="log",log_graph=True)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',save_top_k=-1,save_on_train_epoch_end=True,every_n_train_steps=100)

    trainer = pl.Trainer(
            max_epochs=-1,
            accelerator="gpu",
            devices=4,
            precision=16,
            amp_backend='native',
            benchmark=True,
            logger=logger,
            log_every_n_steps=25,
            callbacks=[checkpoint_callback],
            strategy=pl.strategies.BaguaStrategy(algorithm="async"))

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
