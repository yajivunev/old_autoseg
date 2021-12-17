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
# logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)
# logging.getLogger('daisy.persistence.mongodb_graph_provider').setLevel(logging.DEBUG)
#logging.getLogger('daisy').setLevel(logging.DEBUG)

def agglomerate(
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
        merge_function,
        **kwargs):

    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.
    Args:
        file_name (``string``):
            The input file containing affs and fragments.
        affs_dataset, fragments_dataset (``string``):
            Where to find the affinities and fragments.
        block_size (``tuple`` of ``int``):
            The size of one block in world units.
        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.
        num_workers (``int``):
            How many blocks to run in parallel.
        merge_function (``string``):
            Symbolic name of a merge function. See dictionary below.
    '''

    affs_file = fragments_file = os.path.abspath(
            os.path.join(
                base_dir,experiment,"01_data",setup,str(iteration),file_name
                )
            )

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    network_dir = os.path.join(experiment, setup, str(iteration), merge_function)

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(fragments_file, fragments_dataset, mode='r')

    block_directory = os.path.join(fragments_file, 'block_nodes')

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)

    read_roi = daisy.Roi((0,)*affs.roi.dims(), block_size).grow(context, context)
    write_roi = daisy.Roi((0,)*affs.roi.dims(), block_size)

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            block_directory,
            write_roi,
            merge_function,
            network_dir),
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
        merge_function,
        network_dir,
        **kwargs):

    worker_id = daisy.Context.from_env().worker_id

    logging.info("worker %s started...", worker_id)

    output_dir = os.path.join('.agglomerate_blockwise', network_dir)

    try:
        os.makedirs(output_dir)
    except:
        pass

    log_out = os.path.join(output_dir, 'agglomerate_blockwise_{}.out'.format(worker_id))
    log_err = os.path.join(output_dir, 'agglomerate_blockwise_{}.err'.format(worker_id))

    config = {
            'affs_file': affs_file,
            'affs_dataset': affs_dataset,
            'fragments_file': fragments_file,
            'fragments_dataset': fragments_dataset,
            'block_directory': block_directory,
            'write_size': write_roi.get_shape(),
            'merge_function': merge_function
        }

    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_dir, '%d.config'%config_hash)

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = 'workers/agglomerate_worker.py'

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%worker_id

    command = ["python -u",os.path.join('.', worker),os.path.abspath(config_file)]

    daisy.call(
        command,
        log_out=log_out,
        log_err=log_err)

def check_block(blocks_agglomerated, block):

    done = blocks_agglomerated.count({'block_id': block.block_id}) >= 1

    return done

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    agglomerate(**config)

    end = time.time()

    seconds = end - start
    minutes = seconds/60
    hours = minutes/60
    days = hours/24

    print('Total time to agglomerate: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))
