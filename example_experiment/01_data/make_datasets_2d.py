import daisy
import numpy as np
import sys
import os
import multiprocessing as mp


def write_section(input_zarr,dataset,section,roi,vs,write_roi,write_vs,dtype,idx):

    section_number = int(roi.offset[0]/vs[0] + idx)
    
    if np.any(section):

        print(f"at section {section_number}..")

        new_ds = daisy.prepare_ds(
                input_zarr,
                f"2d_{dataset}/{section_number}",
                write_roi,
                write_vs,
                dtype)

        new_ds[write_roi] = section

    else:
        print(f"section {section_number} is empty, skipping")
        pass


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    dataset = sys.argv[2]

    print(f"making {dataset} 2d..")
    ds = daisy.open_ds(input_zarr,dataset)

    ds_data = ds.to_ndarray()

    roi = ds.roi
    vs = ds.voxel_size
    dtype = ds.dtype
 
    write_roi = daisy.Roi(roi.offset[1:],roi.shape[1:])
    write_vs = vs[1:]

    with mp.Pool(16) as pool:

        pool.starmap(write_section,[(input_zarr,dataset,section,roi,vs,write_roi,write_vs,dtype,idx) for idx,section in enumerate(ds_data)])
