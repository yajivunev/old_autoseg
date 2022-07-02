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
    pred_lsd = gp.ArrayKey('PRED_LSD')
    pred_mask = gp.ArrayKey('PRED_MASK')

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_lsd, output_size)
    chunk_request.add(pred_mask, output_size)

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
                0: pred_lsd,
                1: pred_mask
            },
            checkpoint=os.path.join(setup_dir, 'model_checkpoint_%d'%iteration)
        )
 
    pipeline += gp.Scan(chunk_request)

    pipeline += gp.Squeeze([raw, pred_lsd, pred_mask])

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.shape)
    predict_request.add(pred_lsd, total_output_roi.shape)
    predict_request.add(pred_mask, total_output_roi.shape)

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

        return batch[pred_lsd].data,batch[pred_mask].data


if __name__ == "__main__":

    iteration = int(sys.argv[1])
    raw_file = sys.argv[2]
    raw_dataset = '2d_raw'
    out_file = sys.argv[3]

    raw = daisy.open_ds(
            raw_file,
            'raw')
    
    voxel_size = raw.voxel_size
    sections = raw.shape[0]

    # voxels
    input_shape_3d = gp.Coordinate((1,) + tuple(config['input_shape']))
    output_shape_3d = gp.Coordinate((1,) + tuple(config['output_shape']))

    # nm
    input_size_3d = input_shape_3d*voxel_size
    output_size_3d = output_shape_3d*voxel_size
    context_3d = (input_size_3d - output_size_3d) / 2

    total_roi = raw.roi.grow(-context_3d,-context_3d)

    out_lsd = daisy.prepare_ds(
            out_file,
            'pred_lsd',
            total_roi,
            voxel_size,
            num_channels=6,
            dtype=np.float32)

    print(((total_roi.shape/voxel_size)[0],6) + (total_roi.shape/voxel_size)[-2:])

    pred_lsd = np.zeros(shape=((total_roi.shape/voxel_size)[0],6) + (total_roi.shape/voxel_size)[-2:])

    out_mask = daisy.prepare_ds(
            out_file,
            'pred_mask',
            total_roi,
            voxel_size,
            dtype=np.float32)

    pred_mask = np.zeros(shape=total_roi.shape/voxel_size)

    for z in range(sections):

        raw_ds = raw_dataset + f"/{z}"

        pred_lsd[z],pred_mask[z] = predict(
                iteration,
                raw_file,
                raw_ds)

    out_lsd[total_roi] = np.swapaxes(pred_lsd,0,1)
    out_mask[total_roi] = pred_mask
