import zarr
import numpy as np
import daisy
import sys


def threshold_array(array,threshold):

    array[array >= threshold] = 1
    array[array < threshold] = 0

    return array

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    pred_ds = sys.argv[2]
    thresholds = sys.argv[3:]

    pred = daisy.open_ds(input_zarr,pred_ds)

    pred_data = pred.to_ndarray()

    voxel_size = pred.voxel_size
    roi = pred.roi

    for threshold in thresholds:

        threshold = int(threshold)

        out_ds = daisy.prepare_ds(input_zarr,f"mask_{threshold}",roi,voxel_size,dtype=np.uint8)

        out_array = threshold_array(pred_data,threshold)

        #add more skimage postprocessing

        out_ds[roi] = out_array
