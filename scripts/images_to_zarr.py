import zarr
from PIL import Image
import sys
import numpy as np
import glob
from skimage.exposure import equalize_adapthist as Clahe
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
        res_x,
        clahe_switch):

    images = glob.glob(os.path.join(input_dir,"*"))
    images = natural_sort(images)

    print("Reading images...")

    raw = np.stack([np.array(Image.open(x).convert("L")) for x in images])
    
    assert len(raw.shape) == 3, "3D volume has more than one channel"

    if clahe_switch == 1:
        print("doing CLAHE on raw...")

        clahe_raw = np.empty(shape=raw.shape)
        for i in range(len(raw)):
            clahe_raw[i] = Clahe(raw[i])
            clahe_raw[i] = (255*clahe_raw[i]/np.max(clahe_raw[i])).astype(np.uint8)

        clahe_raw = clahe_raw.astype(np.uint8)
        raw = clahe_raw

    print("Writing to Zarr...")

    output_zarr = zarr.open(output_zarr,"a")

    ds_out = output_zarr.create_dataset(
                "volumes/raw",
                data=raw,
                compressor=zarr.get_codec(
                    {'id': 'gzip', 'level': 5}
                ))
    ds_out.attrs['offset'] = [0,0,0]
    ds_out.attrs['resolution'] = [res_z,res_y,res_x]

if __name__ == "__main__":

    input_dir = str(sys.argv[1]) #path to directory containing TIFF images
    output_zarr = str(sys.argv[2]) #path to output zarr
    clahe_switch = int(sys.argv[3]) #CLAHE switch
    res_z = int(sys.argv[4])
    res_y = int(sys.argv[5])
    res_x = int(sys.argv[6])

    images_to_zarr(
            input_dir,
            output_zarr,
            res_z,
            res_y,
            res_x,
            clahe_switch)
