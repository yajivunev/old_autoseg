import sys
import zarr
import numpy as np

""" Script to take cutout of a large zarr directory. """

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to zarr directory
    output_zarr = str(sys.argv[2]) #path to output zarr
    z_start = int(sys.argv[3])
    y_start = int(sys.argv[4])
    x_start = int(sys.argv[5])
    z_shape = int(sys.argv[6])
    y_shape = int(sys.argv[7])
    x_shape = int(sys.argv[8])

    input_zarr = zarr.open(input_zarr,"r")
    output_zarr = zarr.open(output_zarr,"w")

    print("reading input zarr datasets...")

    raw = input_zarr['clahe_raw/s0']
    labels = input_zarr['glia/s0']
    mask = input_zarr['labels_mask/s0']
    
    resolution = raw.attrs['resolution']    
    offset = raw.attrs['offset']

    raw = raw[z_start:z_start+z_shape,y_start:ystart+y_shape,x_start:x_start+x_shape]
    labels = labels[z_start:z_start+z_shape,y_start:y_start+y_shape,x_start:x_start+x_shape]
    mask = mask[z_start:z_start+z_shape,y_start:y_start+y_shape,x_start:x_start+x_shape]

    print("writing output zarr...")

    for ds_name, data in [
            ('clahe_raw',raw),('glia',labels),('labels_mask',mask)]:

        ds_out = output_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

