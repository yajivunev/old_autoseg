import daisy
import numpy as np
from PIL import Image
import sys
import os
from multiprocessing import Pool


""" Script to save dataset as images. """


def write_image(
        idx,
        section,
        base,
        out_dir):

    img = Image.fromarray(section)

    path = os.path.join(out_dir,str(idx).zfill(3)+'_'+base+".tiff")

    img.save(path)
    print(path)

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    input_ds = sys.argv[2]
    output_dir = sys.argv[3]

    try:
        img_base_name = sys.argv[4]
    except:
        print(f"Using '{input_ds}' as image base name..")
        img_base_name = input_ds

    ds = daisy.open_ds(input_zarr,input_ds)

    try:
        os.makedirs(output_dir)
    except:
        pass

    arr = ds.to_ndarray()

    with Pool(8) as pool:

        pool.starmap(
                write_image,
                [(idx,section,img_base_name,output_dir) for idx,section in enumerate(arr,start=1)]
                )
