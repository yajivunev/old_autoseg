import zarr
from PIL import Image
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

    output_zarr = zarr.open(output_zarr,"a")

    ds_out = output_zarr.create_dataset(
                'raw',
                data=raw,
                compressor=zarr.get_codec(
                    {'id': 'gzip', 'level': 5}
                ))
    ds_out.attrs['offset'] = [0,0,0]
    ds_out.attrs['resolution'] = [res_z,res_y,res_x]

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
