from __future__ import print_function
import json
import logging
import numpy as np
import os
import pymongo
import sys

from gunpowder import *
from gunpowder.torch import Predict

from model import MtlsdModel

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# voxels
input_shape = Coordinate(tuple(config['input_shape']))
output_shape = Coordinate(tuple(config['output_shape']))

# nm
voxel_size = Coordinate(tuple(config['voxel_size']))
input_size = input_shape*voxel_size
output_size = output_shape*voxel_size

def predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file,
        out_dataset,
        worker_config,
        **kwargs):

    model = MtlsdModel(
            config['in_channels'],
            config['num_fmaps'],
            config['fmap_inc_factor'],
            config['downsample_factors'],
            config['kernel_size_down'],
            config['kernel_size_up'])

    model.eval()

    raw = ArrayKey('RAW')
    lsds = ArrayKey('LSDS')
    affs = ArrayKey('AFFS')

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(lsds, output_size)
    chunk_request.add(affs, output_size)

    pipeline = ZarrSource(
            raw_file,
            datasets = {
                raw: raw_dataset
            },
            array_specs = {
                raw: ArraySpec(interpolatable=True),
            }
        )

    pipeline += Pad(raw, size=None)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2,-1)

    pipeline += Unsqueeze([raw])
    pipeline += Stack(1) 

    pipeline += Predict(
            model=model,
            inputs={
                'input': raw
            },
            outputs={
                0: lsds,
                1: affs
            },
            checkpoint=os.path.join(setup_dir, 'model_checkpoint_%d'%iteration)
        )

    pipeline += Squeeze([raw])
    pipeline += Squeeze([raw,affs,lsds])

    pipeline += IntensityScaleShift(affs, 255, 0)
    pipeline += IntensityScaleShift(lsds, 255, 0)

    pipeline += ZarrWrite(
            dataset_names={
                affs: 'volumes/affs',
                lsds: 'volumes/lsds'
            },
            output_filename=out_file
        )
    pipeline += PrintProfilingStats(every=10)

    pipeline += DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: 'read_roi',
                affs: 'write_roi',
                lsds: 'write_roi'
            },
            num_workers=worker_config['num_cache_workers'])

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(**run_config)
