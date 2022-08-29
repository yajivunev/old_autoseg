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

    offset = daisy.Coordinate(cutout_offset)
    shape = daisy.Coordinate(cutout_shape)

    input_zarr = zarr.open(input_zarr,"r")
    output_zarr = zarr.open(output_zarr,"w")

    print("reading input zarr datasets...")

    raw = input_zarr['clahe_raw/s1']
    labels = input_zarr['labels/s1']

    raw = daisy.open_ds(input_zarr,"clahe_raw/s1")
    labels = daisy.open_ds(input_zarr,"labels/s1")
    
    vs = raw.voxel_size
    requested_roi = daisy.Roi(offset,shape)

    print("writing output zarr...")

    for ds_name, data in [
            #('raw',raw)]:#,('labels',labels),('labels_mask',mask)]:
            ('clahe_raw',raw),('labels',labels)]:

        print("cutting out %s .." % ds_name)

        data = data.to_ndarray(requested_roi)

        print("writing %s .." % ds_name)

        ds_out = output_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = vs

if __name__ == "__main__":

    input_zarr = sys.argv[1]

    output_zarr = sys.argv[2]

    cutout_offset = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
    cutout_shape = [int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8])]

    cutout_zarr(input_zarr,output_zarr,cutout_offset,cutout_shape)
