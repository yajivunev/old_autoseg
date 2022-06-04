import daisy
import numpy as np
import sys
import os
import multiprocessing as mp


def write_section(input_zarr,dataset,section,write_roi,write_vs,dtype,idx):
   
    print(f"at section {idx}..")

    new_ds = daisy.prepare_ds(
            input_zarr,
            f"2d_{dataset}/{idx}",
            write_roi,
            write_vs,
            dtype)

    new_ds[write_roi] = section


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    dataset = sys.argv[2]

    print(f"making {dataset} 2d..")
    ds = daisy.open_ds(input_zarr,dataset)

    ds_data = ds.to_ndarray()

    roi = ds.roi
    voxel_size = ds.voxel_size
    dtype = ds.dtype
 
    write_roi = daisy.Roi((0,0),roi.shape[1:])
    write_vs = voxel_size[1:]

    with mp.Pool(2) as pool:

        pool.starmap(write_section,[(input_zarr,dataset,section,write_roi,write_vs,dtype,idx) for idx,section in enumerate(ds_data)])
#    for id,sec in enumerate(ds_data):
#
#        if id % 50 == 0:
#            print(f"at section {id}..")
#
#        new_ds = daisy.prepare_ds(
#                input_zarr,
#                f"2d_{dataset}/{id}",
#                write_roi,
#                write_vs,
#                dtype)
#
#        new_ds[write_roi] = sec
#        #new_ds[write_roi] = np.expand_dims(sec,axis=0)
