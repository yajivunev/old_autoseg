import zarr
import numpy as np
import daisy
import sys

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]
    glia_id = int(sys.argv[3])

    labels = daisy.open_ds(input_zarr,labels_ds)

    glia = labels.to_ndarray()
    glia[glia != glia_id] = 0

    voxel_size = labels.voxel_size
    roi = labels.roi

    glia_ds = daisy.prepare_ds(input_zarr,'glia',roi,voxel_size,dtype=np.uint8)

    glia_ds[roi] = glia
