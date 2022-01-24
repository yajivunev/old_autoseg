import sys
import zarr
import numpy as np

""" Script to make unlabelled volume for a sparsely labeled dataset. """

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to zarr directory
    labels_ds = str(sys.argv[2]) #labels dataset name

    input_zarr = zarr.open(input_zarr,"r+")

    print("reading input zarr datasets...")

    labels = input_zarr[labels_ds]
    
    resolution = labels.attrs['resolution']    
    offset = labels.attrs['offset']

    print("Unlabelling...")

    unlabelled = np.array(labels)
    unlabelled[unlabelled > 0] = 1

    print("writing to zarr...")

    for ds_name, data in [
            ('unlabelled', unlabelled)]:

        ds_out = input_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

