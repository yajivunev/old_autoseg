import daisy
import numpy as np
import sys
import os
import itertools


""" Script to do a bounding-box crop of a dataset. """


def bbox(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    in_dataset = sys.argv[2]
    out_dataset = sys.argv[3]
   
    in_ds = daisy.open_ds(input_zarr,in_dataset)

    in_array = in_ds.to_ndarray()

    print(f"{in_dataset}'s shape: {in_array.shape}")

    bounds = bbox(in_array)
    arr_offset = [bounds[0],bounds[2],bounds[4]]
    arr_shape = [bounds[1]-bounds[0]+1,bounds[3]-bounds[2]+1,bounds[5]-bounds[4]+1]

    print(f"{out_dataset}'s shape: {arr_shape}, offset: {arr_offset}")

    voxel_size = daisy.Coordinate(in_ds.voxel_size)
    
    try:
        output_zarr = sys.argv[4]
        roi = daisy.Roi(daisy.Coordinate([0,0,0]),(daisy.Coordinate(arr_shape)*voxel_size))
    except:
        output_zarr = input_zarr
        print(f"No out_file given, new dataset will be written to {input_zarr}")
        roi = daisy.Roi((daisy.Coordinate(arr_offset)*voxel_size),(daisy.Coordinate(arr_shape)*voxel_size))

    out_ds = daisy.prepare_ds(
            output_zarr,
            out_dataset,
            roi,
            voxel_size,
            dtype=in_ds.dtype)

    out_ds[roi] = in_array[
            bounds[0]:bounds[1]+1,
            bounds[2]:bounds[3]+1,
            bounds[4]:bounds[5]+1]
