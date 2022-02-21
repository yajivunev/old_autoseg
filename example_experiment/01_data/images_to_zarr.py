from PIL import Image
import daisy
import sys
import numpy as np
import glob
import os
import re

""" Script to write directory of images to zarr directory. Perform CLAHE if specified. """

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def images_to_zarr(
        input_dir,
        output_zarr,
        res_z,
        res_y,
        res_x):

    images = glob.glob(os.path.join(input_dir,"*"))
    images = natural_sort(images)

    print("Reading images...")

    raw = np.stack([np.array(Image.open(x).convert("L")) for x in images])
    
    assert len(raw.shape) == 3, "3D volume has more than one channel"

    print("Writing to Zarr...")

    voxel_size = daisy.Coordinate([res_z,res_y_res_x])
    raw_shape = daisy.Coordinate(raw.shape)

    roi = daisy.Roi(([0,]*3),(raw_shape*voxel_size))

    ds_out = daisy.prepare_ds(
                output_zarr,
                'raw',
                roi,
                voxel_size,
                dtype=np.uint8)

    ds_out[roi] = raw

if __name__ == "__main__":

    input_dir = str(sys.argv[1]) #path to directory containing TIFF images
    output_zarr = str(sys.argv[2]) #path to output zarr
    res_z = int(sys.argv[3])
    res_y = int(sys.argv[4])
    res_x = int(sys.argv[5])

    images_to_zarr(
            input_dir,
            output_zarr,
            res_z,
            res_y,
            res_x)
