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

    unlabelled = unlabelled.astype(bool)

    footprint = skimage.morphology.disk(radius=5,decomposition='sequence')

    for section in range(len(unlabelled)):

        print(f"Closing {section}")
        
        skimage.morphology.binary_closing(
                unlabelled[section],
                footprint=footprint,
                out=unlabelled[section])

        skimage.morphology.remove_small_holes(
                unlabelled[section],
                area_threshold=1600,
                out=unlabelled[section])
    
    unlabelled_ds = daisy.prepare_ds(input_zarr,'unlabelled',roi,voxel_size,dtype=np.uint8)

    unlabelled_ds[roi] = unlabelled.astype(np.uint8)
