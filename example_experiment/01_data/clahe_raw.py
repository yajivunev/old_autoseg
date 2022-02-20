import sys
import numpy as np
import daisy
from skimage.exposure import equalize_adapthist as clahe

""" Script to perform CLAHE on volumes/raw in a zarr directory. """

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to zarr directory
    raw_ds = str(sys.argv[2]) #raw dataset name

    print(f"reading input zarr datasets... {raw_ds} in {input_zarr}")

    raw = daisy.open_ds(input_zarr,raw_ds)
   
    resolution = raw.voxel_size
    roi = raw.roi

    print("doing CLAHE on raw...")

    clahe_raw = np.empty(shape=raw.shape)
    
    for i in range(raw.shape[0]):
        clahe_raw[i] = clahe(
                raw.data[i],
                kernel_size=128,
                clip_limit=0.007,
                rescale=False)
        clahe_raw[i] = np.round(255*clahe_raw[i]).astype(np.uint8)

    clahe_raw = clahe_raw.astype(np.uint8)

    print("writing output zarr...")

    ds_out = daisy.prepare_ds(
            input_zarr,
            'clahe_'+raw_ds,
            roi,
            resolution,
            dtype=np.uint8)

    ds_out[roi] = clahe_raw
