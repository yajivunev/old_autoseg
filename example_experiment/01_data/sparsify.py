import os
import sys
import json
import numpy as np
import numba as nb
import daisy
import itertools
import skimage
from scale_pyramid import create_scale_pyramid

@nb.njit
def replace_where_not(arr, needle, replace):
    arr = arr.ravel()
    needles = set(needle)
    for idx in range(arr.size):
        if arr[idx] not in needles:
            arr[idx] = replace

def bbox(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


if __name__ == "__main__":
	
    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]
    #crop_name = sys.argv[3]
    #label_ids = list(map(int,sys.argv[4:])) #list of integer unique ids to keep in crop
    sparsity_json = sys.argv[3]

    #load sparsity partitions json
    with open(sparsity_json,"r") as f:
        partitions = json.load(f)

    for partition_number in partitions.keys():    

        partition = partitions[partition_number]

        assert partition != []

        crop_name = f"{os.path.basename(sparsity_json)[7:-5]}_{partition_number}"
        
        label_ids = [x['unique'] for x in partition]

        #open, read inputs
        raw = daisy.open_ds(input_zarr,'clahe_raw')
        labels = daisy.open_ds(input_zarr,labels_ds)
        
        roi = labels.roi
        vs = labels.voxel_size

        labels_array = labels.to_ndarray()

        #mask out ids not in label_ids; get bounding box coordinates
        replace_where_not(labels_array,np.array(label_ids),0)
        bounds = bbox(labels_array)

        #do the crop
        cropped = labels_array[
                bounds[0]:bounds[1],
                bounds[2]:bounds[3],
                bounds[4]:bounds[5]]

        #get ROIs, pad raw ROI by a little bit.
        arr_offset = daisy.Coordinate([bounds[0],bounds[2],bounds[4]]) 
        arr_shape = daisy.Coordinate([bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]])

        labels_roi = daisy.Roi(
                arr_offset*vs,
                arr_shape*vs)

        padding = arr_shape / 10
        padding += daisy.Coordinate((25,200,200))

        raw_roi = daisy.Roi(
                (arr_offset - padding)*vs,
                (arr_shape + padding * 2)*vs)

        labels_roi += roi.offset
        raw_roi += roi.offset

        #prepare output datasets
        out_raw = daisy.prepare_ds(
                input_zarr,
                f'crops/{crop_name}.zarr/clahe_raw',
                raw_roi,
                vs,
                dtype=np.uint8)
        
        out_labels = daisy.prepare_ds(
                input_zarr,
                f'crops/{crop_name}.zarr/{labels_ds}',
                labels_roi,
                vs,
                dtype=np.uint64)
        
        out_labels_mask = daisy.prepare_ds(
                input_zarr,
                f'crops/{crop_name}.zarr/labels_mask',
                labels_roi,
                vs,
                dtype=np.uint8)
        
        out_unlabelled = daisy.prepare_ds(
                input_zarr,
                f'crops/{crop_name}.zarr/unlabelled',
                labels_roi,
                vs,
                dtype=np.uint8)

        #dump ROI and crop name in crop.json
        with open(os.path.join(input_zarr,f'crops/{crop_name}.zarr/crop.json'),"w") as f:
            json.dump({
                "name": crop_name,
                "offset": labels_roi.offset,
                "shape": labels_roi.shape
                },f,indent=4)

        #make unlabelled
        unlabelled = np.copy(cropped)
        unlabelled[unlabelled > 0] = 1

        #make labels_mask
        labels_mask = np.copy(unlabelled)

        footprint = skimage.morphology.disk(radius=20,decomposition='sequence')

        for section in range(len(labels_mask)):

            print(f"Closing {section}")
            
            skimage.morphology.binary_closing(
                    labels_mask[section],
                    footprint=footprint,
                    out=labels_mask[section])
        
#            labels_mask[section] = skimage.morphology.area_closing(
#                    labels_mask[section],
#                    500,
#                    connectivity=2)

        out_raw[raw_roi] = raw[raw_roi]
        out_labels_mask[labels_roi] = labels_mask.astype(np.uint8)
        out_labels[labels_roi] = cropped
        out_unlabelled[labels_roi] = unlabelled.astype(np.uint8)

        #create scale pyramids
        
        for x in ['clahe_raw',labels_ds,'labels_mask','unlabelled']:
            create_scale_pyramid(
                    input_zarr,
                    f'crops/{crop_name}.zarr/{x}',
                    [[1,2,2],[1,2,2]],
                    None)
