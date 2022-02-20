import sys
import zarr
import csv
import logging
import daisy
import numpy as np

""" Script to remove labels provided in a labeled dataset. """

def remove_in_block(
        block,
        labels_list,
        in_ds,
        out_ds):

    logging.info('Fetching data in block %s' %block.read_roi)
    
    removed = in_ds[block.read_roi].to_ndarray()
    labels_list = np.array(labels_list)

    removed[in1d_alternative_2D(removed,labels_list)] = 0

    out_ds[block.write_roi] = removed

def in1d_alternative_2D(npArr, arr):
    idx = np.searchsorted(arr, npArr.ravel())
    idx[idx==len(arr)] = 0
    return arr[idx].reshape(npArr.shape) == npArr

if __name__ == '__main__':
    
    input_zarr = str(sys.argv[1]) #path to zarr directory
    ds_name = str(sys.argv[2]) #dataset name with integer dtype

    print("reading input zarr datasets...")

    in_ds = daisy.open_ds(input_zarr,ds_name)
    
    resolution = in_ds.voxel_size    
    roi = in_ds.roi

    print("Unlabelling given labels...")

    labels_list = [] #insert integer labels to be removed here, or keep every n-th id
    #labels_list = list(np.unique(in_ds.data))
    #labels_list = [id for id in labels_list if id not in labels_list[::3]][2:]

    out_ds = daisy.prepare_ds(
            input_zarr,
            'labels',
            roi,
            resolution,
            dtype=in_ds.dtype)

    task = daisy.Task(
            'LabelRemoveTask',
            roi,
            daisy.Roi((0, 0, 0), (2000, 2000, 2000)),
            daisy.Roi((0, 0, 0), (2000, 2000, 2000)),
            process_function=lambda b: remove_in_block(
                b,
                labels_list,
                in_ds,
                out_ds),
            fit='shrink',
            num_workers=56)

    daisy.run_blockwise([task])
