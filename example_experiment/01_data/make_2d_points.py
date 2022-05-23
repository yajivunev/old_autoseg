import os
import daisy
import numpy as np
import sys
import glob
from skimage.feature import peak_local_max


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    edt_ds = sys.argv[2]
    output_csv_dir = sys.argv[3]

    print("reading EDT ds..")

    sections = glob.glob(os.path.join(input_zarr,edt_ds,"*"))

    for section in sections:

        id = int(section.split('/')[-1])

        sec_ds = daisy.open_ds(input_zarr,edt_ds+f"/{id}")
        sec_data = sec_ds.to_ndarray()

        in_sec_points = peak_local_max(sec_data, min_distance=3)

#        if len(in_sec_points) == 0:
#            with open(os.path.join(output_csv_dir,f"section_{id}.csv"),"a") as f: 
#                pass

#        else:
        for pt in in_sec_points:
            with open(os.path.join(output_csv_dir,f"section_{id}.csv"),"a") as f:
                f.write(f"{str(pt[0])} {str(pt[1])} 1\n")
