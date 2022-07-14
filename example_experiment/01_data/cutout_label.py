import os
import sys
import json
import numpy as np
import daisy
import itertools
from scale_pyramid import create_scale_pyramid


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
    label_id = int(sys.argv[3])

    lookup_file = os.path.join(input_zarr,labels_ds+'/s0','.obj_lookup')

    with open(lookup_file,"r") as f:
        lookup = {int(v):k for k,v in json.load(f).items()} # lookup = {unique_id : string label}

    label_name = lookup[label_id]

    raw = daisy.open_ds(input_zarr,'raw/s0')
    labels = daisy.open_ds(input_zarr,labels_ds+'/s0')
    unlabelled = daisy.open_ds(input_zarr,'unlabelled/s0')
    
    roi = labels.roi
    vs = labels.voxel_size

    labels_array = labels.to_ndarray()

    bounds = bbox(labels_array==label_id)

    arr_offset = daisy.Coordinate([bounds[0],bounds[2],bounds[4]]) 
    arr_shape = daisy.Coordinate([bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]])

    labels_roi = daisy.Roi(
            arr_offset*vs,
            arr_shape*vs)

    padding = arr_shape / 10
    padding += daisy.Coordinate((0,100,100))

    raw_roi = daisy.Roi(
            (arr_offset - padding)*vs,
            (arr_shape + padding * 2)*vs)

    labels_roi += roi.offset
    raw_roi += roi.offset

    out_raw = daisy.prepare_ds(
            input_zarr,
            f'objects/{label_name}/raw',
            raw_roi,
            vs,
            dtype=np.uint8)
    
    out_labels = daisy.prepare_ds(
            input_zarr,
            f'objects/{label_name}/{labels_ds}',
            labels_roi,
            vs,
            dtype=np.uint64)
    
    out_mask = daisy.prepare_ds(
            input_zarr,
            f'objects/{label_name}/mask',
            labels_roi,
            vs,
            dtype=np.uint8)
    
    out_unlabelled = daisy.prepare_ds(
            input_zarr,
            f'objects/{label_name}/unlabelled',
            labels_roi,
            vs,
            dtype=np.uint8)

    cropped = labels_array[
            bounds[0]:bounds[1],
            bounds[2]:bounds[3],
            bounds[4]:bounds[5]]

    cropped[cropped != label_id] = 0 #mask out everything but label_id

    out_raw[raw_roi] = raw[raw_roi]
    out_mask[labels_roi] = cropped
    out_labels[labels_roi] = labels[labels_roi]
    out_unlabelled = unlabelled[labels_roi]

    #create scale pyramids
    
    for x in ['raw',labels_ds,'mask','unlabelled']:
        create_scale_pyramid(
                input_zarr,
                f'objects/{label_name}/{x}',
                [[1,2,2],[1,2,2]],
                None)
