import os
import sys
import daisy
import numpy as np
import zarr

''' Script to combine datasets into a new mask in a zarr container. '''


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    out_ds = sys.argv[2]
    ds_1 = sys.argv[3]
    ds_2 = sys.argv[4]

    #init with first dataset

    ds_1 = daisy.open_ds(input_zarr,ds_1)
    ds_2 = daisy.open_ds(input_zarr,ds_2)

    vs = ds_1.voxel_size
    roi = ds_1.roi.union(ds_2.roi)
    
    out = daisy.prepare_ds(
            input_zarr,
            out_ds,
            roi,
            vs,
            dtype=np.uint8)

    out[ds_1.roi] = ds_1.to_ndarray()
    out[ds_2.roi] = np.clip(out[ds_2.roi].to_ndarray() + ds_2.to_ndarray(),0,1)
