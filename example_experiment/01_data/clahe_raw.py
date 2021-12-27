import sys
import zarr
import numpy as np
from skimage.exposure import equalize_adapthist as clahe

""" Script to perform CLAHE on volumes/raw in a zarr directory. """

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to zarr directory
    raw_ds = str(sys.argv[2]) #raw dataset name

    input_zarr = zarr.open(input_zarr,"r+")

    print("reading input zarr datasets...")

    raw = input_zarr[raw_ds]
    
    resolution = raw.attrs['resolution']    
    offset = raw.attrs['offset']

    print("doing CLAHE on raw...")

    clahe_raw = np.empty(shape=raw.shape)
    for i in range(len(raw)):
        clahe_raw[i] = clahe(raw[i])
        clahe_raw[i] = (255*clahe_raw[i]/np.max(clahe_raw[i])).astype(np.uint8)

    clahe_raw = clahe_raw.astype(np.uint8)

    print("writing output zarr...")

    for ds_name, data in [
            (raw_ds+'_clahe', clahe_raw[:])]:

        ds_out = input_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

