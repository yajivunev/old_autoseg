import sys
import json
import numpy as np
import logging
import itertools
import time

import tqdm
import daisy
from multiprocessing import Pool

from post import post_lsds

def run_val(args):

    raw_container = args['raw_container']
    pred_container = args['pred_container']
    roi = args['roi']
    downsampling_mode = args['downsampling_mode']
    factor = args['factor']
    target_dwt = args['target_dwt']
    components = args['components']
    normalize_lsds = args['normalize_lsds']
    affs_nb = args['affs_nb']
    affs_max_dist = args['affs_max_dist']
    affs_max_filter = args['affs_max_filter']
    fragments_dwt = args['fragments_dwt']
    bg_mask = args['bg_mask']
    merge_function = args['merge_function']

    results = post_lsds(
        raw_container,
        pred_container,
        roi,
        downsampling_mode,
        factor,
        target_dwt,
        components,
        normalize_lsds,
        affs_nb,
        affs_max_dist,
        affs_max_filter,
        fragments_dwt,
        bg_mask,
        merge_function)

    return args | results["best"]


if __name__ == "__main__":

    grid = sys.argv[1]
    results_out = sys.argv[2]

    with open(grid,"r") as f:
        grid = json.load(f)

    keys, values = zip(*grid.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    length = len(arguments)

    #arguments = arguments[:2]

    results = {}

    print(f"total number of validation runs: {length}")
    
    with Pool(8) as pool:

        for i,result in enumerate(tqdm.tqdm(pool.imap(run_val,arguments,chunksize=1),total=length)):
                results[i] = result

    if results != {}:
        with open(results_out,'w') as f:
            json.dump(results,f,indent=4)
