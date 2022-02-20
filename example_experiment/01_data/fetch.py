import numpy as np
import daisy
import h5py
import zarr
from cloudvolume import CloudVolume
import sys

""" Script to fetch dataset from cloud source and write to Zarr/HDF5. """

raw_vol_url = "s3://open-neurodata/kharris15/apical/em"
labels_vol_url = "s3://open-neurodata/kharris15/apical/anno"

raw_vol = CloudVolume(raw_vol_url, mip=0, progress=True, use_https=True)

labels_vol = CloudVolume(labels_vol_url, mip=0, bounded=True, progress=True, use_https=True, fill_missing=True)

labels_offset = [60,3241,3493]
labels_end = [160,4815,4978]

context = [20,500,500]

raw_offset = [i-j for i,j in zip(labels_offset, context)]
raw_end = [i+j for i,j in zip(labels_end, context)]

labels_data = labels_vol[
        labels_offset[2]:labels_end[2],
        labels_offset[1]:labels_end[1],
        labels_offset[0]:labels_end[0]]

raw_data = raw_vol[
        raw_offset[2]:raw_end[2],
        raw_offset[1]:raw_end[1],
        raw_offset[0]:raw_end[0]]

#convert to ZYX
raw_data = np.array(np.transpose(raw_data))[0,...]

labels_data = np.array(np.transpose(labels_data))[0,...]

voxel_size = raw_vol.info['scales'][0]['resolution']
voxel_size = daisy.Coordinate(voxel_size[::-1])

raw_offset = daisy.Coordinate(raw_offset)*voxel_size
raw_shape = daisy.Coordinate(raw_data.shape)*voxel_size
labels_offset = daisy.Coordinate(labels_offset)*voxel_size
labels_shape = daisy.Coordinate(labels_data.shape)*voxel_size

raw_roi = daisy.Roi((raw_offset),(raw_shape))
labels_roi = daisy.Roi((labels_offset),(labels_shape))

raw_ds = daisy.prepare_ds(
    'apical.zarr',
    'raw',
    raw_roi,
    voxel_size,
    dtype=np.uint8)

raw_ds[raw_roi] = raw_data

labels_ds = daisy.prepare_ds(
    'apical.zarr',
    'labels_from_cloud',
    labels_roi,
    voxel_size,
    dtype=np.uint64)

labels_ds[labels_roi] = labels_data
