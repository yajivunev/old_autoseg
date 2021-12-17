import sys
import zarr
import numpy as np
from skimage.exposure import equalize_adapthist as clahe

""" Script to perform CLAHE on volumes/raw in a zarr directory. """

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to input zarr directory
    output_zarr = str(sys.argv[2]) #path to output zarr directory

    input_zarr = zarr.open(input_zarr,"r")
    output_zarr = zarr.open(output_zarr,"w")

    print("reading input zarr datasets...")

    raw = input_zarr['volumes/raw']
    neuron_ids = input_zarr['volumes/labels/neuron_ids']
    mask = input_zarr['volumes/labels/mask']
    
    assert raw.shape == neuron_ids.shape == mask.shape, "dataset shapes are not the same"
    assert raw.attrs['resolution'] == neuron_ids.attrs['resolution'] == mask.attrs['resolution'], "resolutions are not the same."
    assert raw.attrs['offset'] == neuron_ids.attrs['offset'] == mask.attrs['offset'], "offsets are not the same."
    
    resolution = raw.attrs['resolution']    
    offset = raw.attrs['offset']

    print("doing CLAHE on raw...")

    clahe_raw = np.empty(shape=raw.shape)
    for i in range(len(raw)):
        clahe_raw[i] = clahe(raw[i])
        clahe_raw[i] = (255*clahe_raw[i]/np.max(clahe_raw[i])).astype(np.uint8)

    clahe_raw = clahe_raw.astype(np.uint8)

    print("writing output zarr...")

    for ds_name, data in [
            ('volumes/raw', clahe_raw[:]),
            ('volumes/labels/neuron_ids', neuron_ids[:])]:,
            ('volumes/labels/mask', mask[:])]:

        ds_out = output_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

