import sys
import numpy as np
import daisy

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    ds_names = sys.argv[2:]

    for name in ds_names:

        ds = daisy.open_ds(input_zarr,name)

        unique = list(np.unique(ds.to_ndarray()))

        print(name, unique, len(unique))
