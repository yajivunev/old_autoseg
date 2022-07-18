import zarr
import numpy as np
import daisy
import sys
import skimage.morphology


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]

    labels = daisy.open_ds(input_zarr,labels_ds)

    voxel_size = labels.voxel_size
    roi = labels.roi

    labels_arr = labels.to_ndarray()
    
    #make unlabelled
    unlabelled = np.copy(labels_arr)
    unlabelled[unlabelled > 0] = 1

    unlabelled_ds = daisy.prepare_ds(input_zarr,'unlabelled',roi,voxel_size,dtype=np.uint8)

    unlabelled_ds[roi] = unlabelled

    #make labels_mask
    labels_mask = np.copy(unlabelled)

    footprint = skimage.morphology.disk(radius=20,decomposition='sequence')

    for section in range(len(labels_mask)):

        print(f"Closing {section}")
        
        skimage.morphology.binary_closing(
                labels_mask[section],
                footprint=footprint,
                out=labels_mask[section])
    
        labels_mask[section] = skimage.morphology.area_closing(
                labels_mask[section],
                64000,
                connectivity=2)
    
    labels_mask_ds = daisy.prepare_ds(input_zarr,'labels_mask',roi,voxel_size,dtype=np.uint8)

    labels_mask_ds[roi] = labels_mask
