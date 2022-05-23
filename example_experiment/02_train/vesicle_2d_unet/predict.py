import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import json
import torch
import zarr

from model import UnetModel


setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# voxels
input_shape = gp.Coordinate(tuple(config['input_shape']))
output_shape = gp.Coordinate(tuple(config['output_shape']))

# nm
net_voxel_size = gp.Coordinate(tuple(config['voxel_size']))
input_size = input_shape*net_voxel_size
output_size = output_shape*net_voxel_size
context = (input_size - output_size) / 2

def predict(
        iteration,
        raw_file,
        raw_dataset):


    model = UnetModel(
            config['in_channels'],
            config['num_fmaps'],
            config['fmap_inc_factor'],
            config['downsample_factors'],
            config['kernel_size_down'],
            config['kernel_size_up'])

    model.eval()

    raw = gp.ArrayKey('RAW')
    raster = gp.ArrayKey('RASTER')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(raster, output_size)

    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=net_voxel_size)
            }
        )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    model.eval()

    pipeline = source
    
    pipeline += gp.Normalize(raw)
    pipeline += gp.Pad(raw,None)

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += gp.torch.Predict(
            model=model,
            inputs={
                'input': raw
            },
            outputs={
                0: raster
            },
            checkpoint=os.path.join(setup_dir, 'model_checkpoint_%d'%iteration)
        )
 
    pipeline += gp.Scan(chunk_request)

    pipeline += gp.Squeeze([raw, raster])
    pipeline += gp.Squeeze([raw, raster])

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(raster, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

        return batch[raster].data


if __name__ == "__main__":

    iteration = int(sys.argv[1])
    raw_file = sys.argv[2]
    raw_dataset = '2d_raw'
    out_file = sys.argv[3]
    out_dataset = 'raster'

    raw = daisy.open_ds(
            raw_file,
            'raw')

    sections = raw.shape[0]
    total_roi = raw.roi
    voxel_size = raw.voxel_size

    out = daisy.prepare_ds(
            out_file,
            out_dataset,
            total_roi,
            voxel_size,
            dtype=np.float32)

    pred = np.zeros(shape=raw.shape)

    for z in range(sections):

        raw_ds = raw_dataset + f"/{z}"

        pred[z] = predict(
                iteration,
                raw_file,
                raw_ds)

    out[total_roi] = pred
