import os
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
import gc

def run_val(args):

    raw_container = args['raw_container']
    pred_container = args['pred_container']
    roi = args['roi']
    downsampling = args['downsampling']
    lsds_dwt = args['lsds_dwt']
    components = args['components']
    normalize_lsds = args['normalize_lsds']
    affs_nb = args['affs_nb']
    affs_max_filter = args['affs_max_filter']
    fragments_dwt = args['fragments_dwt']
    bg_mask = args['bg_mask']
    merge_function = args['merge_function']

    results = post_lsds(
        raw_container,
        pred_container,
        roi,
        downsampling,
        lsds_dwt,
        components,
        normalize_lsds,
        affs_nb,
        affs_max_filter,
        fragments_dwt,
        bg_mask,
        merge_function)

    return args | results


if __name__ == "__main__":

    grid = sys.argv[1]
    results_out = sys.argv[2]

    with open(grid,"r") as f:
        grid = json.load(f)

    keys, values = zip(*grid.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    arguments = arguments[-23:]
    length = len(arguments)

    print(f"total number of validation runs: {length}")
    
    with Pool(128,maxtasksperchild=1) as pool:

        for i,result in enumerate(tqdm.tqdm(pool.imap_unordered(run_val,arguments),total=length)):
            with open(os.path.join(results_out,f"{i+89976}.json"),"w") as f:
                json.dump(result,f,indent=4)
