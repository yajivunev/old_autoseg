import daisy
import logging
import lsd
import json
import sys
import time

logging.getLogger().setLevel(logging.DEBUG)

def extract_fragments_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    context = config['context']
    block_directory = config['block_directory']
    write_size = config['write_size']
    num_voxels_in_block = config['num_voxels_in_block']
    fragments_in_xy=config['fragments_in_xy']
    epsilon_agglomerate=config['epsilon_agglomerate']
    filter_fragments=config['filter_fragments']
    replace_sections=config['replace_sections']

    logging.info("Reading affs from %s", affs_file)
    affs = daisy.open_ds(affs_file, affs_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if config['mask_file']:

        logging.info("Reading mask from %s", config['mask_file'])
        mask = daisy.open_ds(
            config['mask_file'],
            config['mask_dataset'],
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

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            logging.info("block read roi begin: %s", block.read_roi.offset)
            logging.info("block read roi shape: %s", block.read_roi.shape)
            logging.info("block write roi begin: %s", block.write_roi.offset)
            logging.info("block write roi shape: %s", block.write_roi.shape)

            lsd.watershed_in_block(
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

            client.release_block(block)

if __name__ == '__main__':

    extract_fragments_worker(sys.argv[1])
