import json
import daisy
import numpy as np
import os
import sys


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]

    lookup_file = os.path.join(input_zarr,labels_ds,'.obj_lookup')

    with open(lookup_file,"r") as f:
        lookup_json = json.load(f)

    lookup = {int(v):k for k,v in lookup_json.items()} # lookup = {unique_id : string label}

    labels_arr = daisy.open_ds(input_zarr,labels_ds).to_ndarray()

    uniques,counts = np.unique(labels_arr,return_counts=True)

    count_dict = {int(k):int(v) for k,v in zip(counts,uniques)} # count_dict = {voxel count : unique_id}

    sorted_counts = sorted(counts,reverse=True)

    for index,count in enumerate(sorted_counts[1:]):

        unique_id = count_dict[count]
        label = lookup[unique_id]

        print(f"{index}. {label}: {count} , unique_id = {unique_id}")
