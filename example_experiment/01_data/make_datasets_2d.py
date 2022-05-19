import daisy
import numpy as np
import sys
import os


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    datasets_to_convert = sys.argv[2:]

    for dataset in datasets_to_convert:

        print(f"making {dataset} 2d..")
        ds = daisy.open_ds(input_zarr,dataset)

        ds_data = ds.to_ndarray()

        roi = ds.roi
        voxel_size = ds.voxel_size
        dtype = ds.dtype

        for id,sec in enumerate(ds_data):

            if id % 50 == 0:
                print(f"at section {id}..")

            write_roi = daisy.Roi((0,0),roi.shape[1:])
            #write_roi = daisy.Roi((0,0,0),(50,)+roi.shape[1:])
            write_vs = voxel_size[1:]
            #write_vs = voxel_size

            new_ds = daisy.prepare_ds(
                    input_zarr,
                    f"2d_{dataset}/{id}",
                    write_roi,
                    write_vs,
                    dtype)

            new_ds[write_roi] = sec
            #new_ds[write_roi] = np.expand_dims(sec,axis=0)
