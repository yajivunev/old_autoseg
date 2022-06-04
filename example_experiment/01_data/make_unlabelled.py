import zarr
import numpy as np
import daisy
import sys

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]
    out_ds_name = sys.argv[3]

    labels = daisy.open_ds(input_zarr,labels_ds)

    unlabelled = labels.to_ndarray()
    unlabelled[unlabelled > 0] = 1

    voxel_size = labels.voxel_size
    roi = labels.roi

    unlabelled_ds = daisy.prepare_ds(input_zarr,out_ds_name,roi,voxel_size,dtype=np.uint8)

    unlabelled_ds[roi] = unlabelled
