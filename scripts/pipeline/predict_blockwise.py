import hashlib
import json
import logging
import numpy as np
import os
import daisy
import sys
import time
import datetime
import subprocess

#logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

def predict_blockwise(
        base_dir,
        experiment,
        setup,
        iteration,
        raw_dataset,
        file_name,
        num_workers,
        num_cache_workers,
        auto_file=None,
        auto_dataset=None):

    '''Run prediction in parallel blocks. Within blocks, predict in chunks.
    Args:
        base_dir (``string``):
            Path to base directory containing experiment sub directories.
        experiment (``string``):
            Name of the experiment (cremi, fib19, fib25, ...).
        setup (``string``):
            Name of the setup to predict.
        iteration (``int``):
            Training iteration to predict from.
        raw_dataset (``string``):
        auto_file (``string``):
        auto_dataset (``string``):
            Paths to the input autocontext datasets (affs or lsds). Can be None if not needed.
        **Note:
            out_dataset no longer needed as input, build out_dataset from config
        file_name (``string``):
            Name of output file
        num_workers (``int``):
            How many blocks to run in parallel.
    '''

    experiment_dir = os.path.join(base_dir, experiment)
    data_dir = os.path.join(experiment_dir, '01_data')
    train_dir = os.path.join(experiment_dir, '02_train')

    raw_file = os.path.abspath(os.path.join(data_dir,file_name))
    out_file = os.path.abspath(os.path.join(data_dir, setup, str(iteration), file_name))

    setup = os.path.abspath(os.path.join(train_dir, setup))

    # from here on, all values are in world units (unless explicitly mentioned)

    # get ROI of source
    try:
        source = daisy.open_ds(raw_file, raw_dataset)
    except:
        raw_dataset = raw_dataset + '/s0'
        source = daisy.open_ds(raw_file, raw_dataset)
    
    logging.info('Source dataset has shape %s, ROI %s, voxel size %s'%(source.shape, source.roi, source.voxel_size))

    # load config
    with open(os.path.join(setup, 'config.json')) as f:
        logging.info('Reading setup config from %s'%os.path.join(setup, 'config.json'))
        net_config = json.load(f)
    outputs = net_config['outputs']

    # get chunk size and context
    net_input_size = daisy.Coordinate(net_config['input_shape'])*source.voxel_size
    net_output_size = daisy.Coordinate(net_config['output_shape'])*source.voxel_size
    context = (net_input_size - net_output_size)/2
    print('CONTEXT: ', context)

    # get total input and output ROIs
    input_roi = source.roi.grow(context, context)
    output_roi = source.roi

    # create read and write ROI
    block_read_roi = daisy.Roi((0, 0, 0), net_input_size) - context
    block_write_roi = daisy.Roi((0, 0, 0), net_output_size)

    logging.info('Preparing output dataset...')

    for output_name, val in outputs.items():
        out_dims = val['out_dims']
        out_dtype = val['out_dtype']
        out_dataset = 'volumes/%s'%output_name

        ds = daisy.prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_roi=block_write_roi,
            num_channels=out_dims,
            compressor={'id': 'gzip', 'level':5}
            )

    logging.info('Starting block-wise processing...')

    # process block-wise
    task = daisy.Task(
        'PredictBlockwiseTask',
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda b: predict_worker(
            b,
            experiment,
            setup,
            iteration,
            raw_file,
            raw_dataset,
            auto_file,
            auto_dataset,
            out_file,
            out_dataset,
            num_cache_workers),
        check_function = None,
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=5,
        fit='overhang')

    return task

def predict_worker(
        block,
        experiment,
        setup,
        iteration,
        raw_file,
        raw_dataset,
        auto_file,
        auto_dataset,
        out_file,
        out_dataset,
        num_cache_workers):

    setup_dir = os.path.join('..', experiment, '02_train', setup)
    predict_script = os.path.abspath(os.path.join(setup_dir, 'predict.py'))

    if raw_file.endswith('.json'):
        with open(raw_file, 'r') as f:
            spec = json.load(f)
            raw_file = spec['container']
    
    worker_config = {
        'num_cache_workers': num_cache_workers
    }

    config = {
        'iteration': iteration,
        'raw_file': raw_file,
        'raw_dataset': raw_dataset,
        'auto_file': auto_file,
        'auto_dataset': auto_dataset,
        'out_file': out_file,
        'out_dataset': out_dataset,
        'worker_config': worker_config
    }

    # get a unique hash for this configuration
    config_str = ''.join(['%s'%(v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))
    #context_str = os.environ['DAISY_CONTEXT']

    worker_id = int(daisy.Context.from_env()['worker_id'])
    
    try:
        os.makedirs(output_dir)
    except:
        pass

    config_file = os.path.join('.','daisy_logs','PredictBlockwiseTask', '%d.config'%config_hash)

    logging.info('Running block with config %s...'%config_file)

    command = ["python -u",predict_script,os.path.abspath(config_file)]
    config["command"] = ' '.join(command)
    #config["context"] = context_str

    with open(config_file, 'w') as f:
        json.dump(config, f)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%worker_id
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        subprocess.check_call(
            ' '.join(command),
            shell=True)
    except subprocess.CalledProcessError as exc:
        raise Exception(
            "Calling %s failed with return code %s, stderr in %s" %
            (' '.join(command), exc.returncode, sys.stderr.name))
    except KeyboardInterrupt:
        raise Exception("Canceled by SIGINT")

    logging.info('daisy command called')

    # if things went well, remove temporary files
    # os.remove(config_file)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()

    task = predict_blockwise(**config)

    succeeded = daisy.run_blockwise([task])

    if not succeeded:
        raise RuntimeError("Prediction failed for (at least) one block")

    end = time.time()

    seconds = end - start
    logging.info('Total time to predict: %f seconds' % seconds)