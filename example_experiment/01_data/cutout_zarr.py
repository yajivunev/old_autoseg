import sys
import zarr
import daisy
import numpy as np

""" Script to take cutout of a large zarr directory. """

def cutout_zarr(
        input_zarr,
        output_zarr,
        cutout_offset,
        cutout_shape):

    z_start = cutout_offset[0]
    y_start = cutout_offset[1]
    x_start = cutout_offset[2]
    z_shape = cutout_shape[0]
    y_shape = cutout_shape[1]
    x_shape = cutout_shape[2]

    input_zarr = zarr.open(input_zarr,"r")
    output_zarr = zarr.open(output_zarr,"w")

    print("reading input zarr datasets...")

    raw = input_zarr['raw']
    labels = input_zarr['labels']
    mask = input_zarr['labels_mask']

    resolution = raw.attrs['resolution']  
    offset = raw.attrs['offset']
    offset = [
            offset[0]+(z_start*resolution[0]),
            offset[1]+(y_start*resolution[1]),
            offset[2]+(x_start*resolution[2])]

    print("writing output zarr...")

    for ds_name, data in [
            #('raw',raw)]:#,('labels',labels),('labels_mask',mask)]:
            ('raw',raw),('labels',labels),('labels_mask',mask)]:

        print("cutting out %s .." % ds_name)

        data = data[
                z_start:z_start+z_shape,
                y_start:y_start+y_shape,
                x_start:x_start+x_shape]

        print("writing %s .." % ds_name)

        ds_out = output_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

if __name__ == "__main__":

    input_zarr = sys.argv[1]

    output_zarr = sys.argv[2]

    cutout_offset = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
    cutout_shape = [int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8])]

    cutout_zarr(input_zarr,output_zarr,cutout_offset,cutout_shape)
