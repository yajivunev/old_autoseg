import h5py
import numpy as np
import sys
import zarr

""" Script to make zarr directory from raw and gt HDF5's. """

if __name__ == '__main__':

    input_h5_raw = str(sys.argv[1])
    raw_ds = str(sys.argv[2])
    input_h5_gt = str(sys.argv[3])
    gt_ds = str(sys.argv[4])
    output_zarr = str(sys.argv[5]) ##5

    f_in_gt = h5py.File(input_h5_gt, 'r')
    f_in_gm = h5py.File(input_h5_raw, 'r')
    f_out = zarr.open(output_zarr, 'w')

    print("Reading neuron IDs...")

    neuron_ids = f_in_gt[gt_ds][:]

    print("Reading raw...")
    raw = f_in_gm[raw_ds][:]

    print("Creating GT mask...")
    # set to 1 where gt ids are true
    
    mask = (neuron_ids < np.uint64(-10)).astype(np.uint8)

    for ds_name, data in [
            ('volumes/raw', raw),
            ('volumes/labels/neuron_ids', neuron_ids),
            ('volumes/labels/mask', mask)]:

        ds_out = f_out.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = [0,0,0]
        ds_out.attrs['resolution'] = [40,4,4]
