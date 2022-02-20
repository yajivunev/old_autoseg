import sys
import zarr
import csv
import logging
import daisy
import numpy as np

""" Script to move labels of synapse in a labeled dataset to another. """

def remove_in_block(
        block,
        raw,
        in_labels,
        out_labels,
        synapses):

    logging.info('Fetching data in block %s' %block.read_roi)
   
    raw = raw[block.read_roi].to_ndarray()
    labels_data = in_labels[block.read_roi].to_ndarray()

    unique = list(np.unique(labels_data))
    logging.debug(f"unique IDs in block:{block.read_roi} are {unique}")

    for i in unique:

        if i in [0,94,97]:
            continue

        m = np.mean(raw[labels_data==i])

        if m < 100:

            logging.info('Found synapse ID %s' % i)
            
            synapses_data = synapses[block.write_roi].to_ndarray()
            
            synapses_data[labels_data==i] = i
            
            labels_data[labels_data==i] = 0
            
            to_write_labels = daisy.Array(
                    data=labels_data,
                    roi=block.read_roi,
                    voxel_size=in_labels.voxel_size)

            to_write_synapses = daisy.Array(
                    data=synapses_data,
                    roi=block.read_roi,
                    voxel_size=in_labels.voxel_size)
            
            synapses[block.write_roi] = to_write_synapses[block.write_roi]
            out_labels[block.write_roi] = to_write_labels[block.write_roi]


if __name__ == '__main__':
    
    input_zarr = str(sys.argv[1]) #path to zarr directory
    raw_ds = str(sys.argv[2]) #raw ds name
    labels_ds = str(sys.argv[2]) #labels ds name

    print("reading input zarr datasets...")

    raw = daisy.open_ds(input_zarr,raw_ds)
    labels = daisy.open_ds(input_zarr,labels_ds)
    
    resolution = labels.voxel_size    
    roi = labels.roi
    chunk_shape = labels.chunk_shape
    write_size = chunk_shape*resolution
    block_roi =  daisy.Roi((0,0,0),(50,100,100))

    print("Unlabelling given labels...")
    print(write_size)

    synapses = daisy.prepare_ds(
            input_zarr,
            'synapses',
            roi,
            resolution,
            dtype=labels.dtype,
            write_size=write_size)
    
    out_labels = daisy.prepare_ds(
            input_zarr,
            'labels',
            roi,
            resolution,
            dtype=labels.dtype,
            write_size=write_size)

    task = daisy.Task(
            'SynapseRemoveTask',
            roi,
            block_roi,
            block_roi,
            process_function=lambda b: remove_in_block(
                b,
                raw,
                labels,
                out_labels,
                synapses),
            fit='shrink',
            num_workers=896)

    daisy.run_blockwise([task])
