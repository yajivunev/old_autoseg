import json
import hashlib
import logging
import numpy as np
import os
import daisy
import sys
import time
import subprocess

from watershed import watershed_in_block

logging.getLogger().setLevel(logging.INFO)
#logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)

def extract_fragments(
        base_dir,
        experiment,
        setup,
        iteration,
        file_name,
        affs_dataset,
        fragments_dataset,
        block_size,
        context,
        num_workers,
        fragments_in_xy,
        epsilon_agglomerate=0,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0,
        replace_sections=None,
        **kwargs):
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.
    Args:
        affs_dataset,
        block_size (``tuple`` of ``int``):
            The size of one block in world units.
        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.
        num_workers (``int``):
            How many blocks to run in parallel.
    '''
    affs_file = fragments_file = os.path.abspath(
            os.path.join(
                base_dir,experiment,"01_data",setup,str(iteration),file_name
                )
            )

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    fragments_file = affs_file
    block_directory = os.path.join(fragments_file, 'block_nodes')

    os.makedirs(block_directory, exist_ok=True)

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0,0,0), block_size),
        compressor={'id': 'zlib', 'level':5})

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims, block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims, block_size)

    num_voxels_in_block = (write_roi/affs.voxel_size).size

    task = daisy.Task(
        'ExtractFragmentsBlockwiseTask',
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda b: extract_fragments_worker(
            b,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            context,
            block_directory,
            write_roi.shape,
            num_voxels_in_block,
            fragments_in_xy,
            epsilon_agglomerate,
            filter_fragments,
            replace_sections,
            mask_file,
            mask_dataset),
        check_function=None,
        num_workers=num_workers,
        max_retries=7,
        read_write_conflict=False,
        fit='shrink')

    return task


def extract_fragments_worker(
        block,
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        context,
        block_directory,
        write_size,
        num_voxels_in_block,
        fragments_in_xy,
        epsilon_agglomerate,
        filter_fragments,
        replace_sections,
        mask_file,
        mask_dataset):

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if mask_file is not None:

        logging.info("Reading mask from {}".format(mask_file))
        mask = daisy.open_ds(
            mask_file,
            mask_dataset,
            mode='r')

    else:

        mask = None

    # open RAG DB
    logging.info("Opening RAG file...")
    rag_provider = daisy.persistence.FileGraphProvider(
        directory=block_directory,
        chunk_size=write_size,
        mode='r+',
        directed=False,
        position_attribute=['center_z', 'center_y', 'center_x']
        )
    
    logging.info("RAG file opened")

    logging.info("block read roi begin: %s", block.read_roi.offset)
    logging.info("block read roi shape: %s", block.read_roi.shape)
    logging.info("block write roi begin: %s", block.write_roi.offset)
    logging.info("block write roi shape: %s", block.write_roi.shape)

    watershed_in_block(
        affs,
        block,
        context,
        rag_provider,
        fragments,
        num_voxels_in_block=num_voxels_in_block,
        mask=mask,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        filter_fragments=filter_fragments,
        replace_sections=replace_sections)


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    task = extract_fragments(**config)

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("ExtractFragments failed for (at least) one block")

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
