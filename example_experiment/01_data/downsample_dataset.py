import sys
import os
import numpy as np
import daisy

""" Script to downsample a zarr array and all its datasets into a new container. """

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    output_zarr = sys.argv[2]
    factor = int(sys.argv[3]) #single integer, to downsample in X and Y only.
    datasets = sys.argv[4:]

    for dataset in datasets:

        ds = daisy.open_ds(input_zarr,dataset)

        roi = ds.roi
        vs = ds.voxel_size
        arr = ds.to_ndarray()

        downsampling = daisy.Coordinate((1,factor,factor))

        new_arr = arr[::1,::factor,::factor]
        new_vs = vs * downsampling
        new_offset = roi.offset

        if not daisy.Coordinate(new_arr.shape) * new_vs == roi.shape:
            padding = (daisy.Coordinate(new_arr.shape) * new_vs) - roi.shape
            padding = [(int(x/2),int(x/2)) for x in padding]
            print(padding)

            new_arr = np.pad(new_arr,padding)

            #new_offset += daisy.Coordinate([x[0] for x in padding])*new_vs

        new_roi = daisy.Roi((0,0,0),daisy.Coordinate(new_arr.shape)*new_vs)

        print(f"old shape = {arr.shape}. new shape = {new_arr.shape}")
        print(f"old vs = {vs}. new vs = {new_vs}")
        print(f"old roi = {roi}. new roi = {new_roi}")

        out = daisy.prepare_ds(
                output_zarr,
                dataset,
                new_roi,
                new_vs,
                ds.dtype)

        out[new_roi] = new_arr
