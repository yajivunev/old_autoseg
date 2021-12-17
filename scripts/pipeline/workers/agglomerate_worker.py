import daisy
import json
import logging
import lsd
import pymongo
import sys
import time

logging.basicConfig(level=logging.INFO)

def agglomerate_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    merge_function = config['merge_function']
    block_directory = config['block_directory']
    write_size = config['write_size']

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs = daisy.open_ds(affs_file, affs_dataset)

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    #opening RAG file
    logging.info("Opening RAG file...")
    rag_provider = daisy.persistence.FileGraphProvider(
        directory=block_directory,
        chunk_size=write_size,
        mode='r+',
        directed=False,
        edges_collection='edges_' + merge_function,
        position_attribute=['center_z', 'center_y', 'center_x']
        )
    logging.info("RAG file opened")

    client = daisy.Client()

    while True:

        block = client.acquire_block()

        if block is None:
            break

        start = time.time()

        lsd.agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                block,
                merge_function=waterz_merge_function,
                threshold=1.0)

        client.release_block(block, ret=0)


if __name__ == '__main__':

    agglomerate_worker(sys.argv[1])
