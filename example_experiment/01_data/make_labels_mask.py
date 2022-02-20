import zarr
import numpy as np
import daisy
import sys

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]

    labels = daisy.open_ds(input_zarr,labels_ds)

    labels_mask = np.ones_like(labels.data,dtype=np.uint8)

    voxel_size = labels.voxel_size
    roi = labels.roi

    mask_ds = daisy.prepare_ds(input_zarr,'labels_mask',roi,voxel_size,dtype=np.uint8)

    mask_ds[roi] = labels_mask
