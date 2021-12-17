import json
import hashlib
import logging
import lsd
import numpy as np
import os
import daisy
import sys
import time

logging.basicConfig(level=logging.INFO)
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

    network_dir = os.path.join(experiment, setup, str(iteration))

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

    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    num_voxels_in_block = (write_roi/affs.voxel_size).size()

    daisy.run_blockwise(
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            block_directory,
            write_roi,
            context,
            fragments_in_xy,
            network_dir,
            epsilon_agglomerate,
            mask_file,
            mask_dataset,
            filter_fragments,
            replace_sections,
            num_voxels_in_block),
        check_function=None,
        num_workers=num_workers,
        read_write_conflict=False,
        fit='shrink')

def start_worker(
        affs_file,
        affs_dataset,
        fragments_file,
        fragments_dataset,
        block_directory,
        write_roi,
        context,
        fragments_in_xy,
        network_dir,
        epsilon_agglomerate,
        mask_file,
        mask_dataset,
        filter_fragments,
        replace_sections,
        num_voxels_in_block,
        **kwargs):

    worker_id = daisy.Context.from_env().worker_id

    logging.info("worker %s started...", worker_id)

    output_dir = os.path.join('.extract_fragments_blockwise', network_dir)

    logging.info('epsilon_agglomerate: %s', epsilon_agglomerate)
    logging.info('mask_file: %s', mask_file)
    logging.info('mask_dataset: %s', mask_dataset)
    logging.info('filter_fragments: %s', filter_fragments)
    logging.info('replace_sections: %s', replace_sections)

    try:
        os.makedirs(output_dir)
    except:
        pass

    log_out = os.path.join(output_dir, 'extract_fragments_blockwise_{}.out'.format(worker_id))
    log_err = os.path.join(output_dir, 'extract_fragments_blockwise_{}.err'.format(worker_id))

    config = {
            'affs_file': affs_file,
            'affs_dataset': affs_dataset,
            'fragments_file': fragments_file,
            'fragments_dataset': fragments_dataset,
            'context': context,
            'block_directory': block_directory,
            'write_size': write_roi.get_shape(),
            'fragments_in_xy': fragments_in_xy,
            'epsilon_agglomerate': epsilon_agglomerate,
            'mask_file': mask_file,
            'mask_dataset': mask_dataset,
            'filter_fragments': filter_fragments,
            'replace_sections': replace_sections,
            'num_voxels_in_block': num_voxels_in_block
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.json'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = 'workers/extract_fragments_worker.py'

    command = ["python -u",os.path.join(".", worker),os.path.abspath(config_file)]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%worker_id

    daisy.call(
        command,
        log_out=log_out,
        log_err=log_err)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    extract_fragments(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
