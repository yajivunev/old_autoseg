import os
import daisy
import json
import sys
import time
from predict_blockwise import predict_blockwise
from extract_fragments import extract_fragments
from agglomerate_blockwise import agglomerate
from find_segments import find_segments
from extract_segmentation_from_lut import extract_segmentation

""" Run pipeline blockwise. """


if __name__ == "__main__":

    config_path = os.path.abspath(sys.argv[1])
    
    config01 = os.path.join(config_path,'01_predict.json')
    config02 = os.path.join(config_path,'02_fragments.json')
    config03 = os.path.join(config_path,'03_agglomerate.json')
    config04 = os.path.join(config_path,'04_find_segments.json')
    config05 = os.path.join(config_path,'05_segmentation.json')

    with open(config01, 'r') as f:
        config01 = json.load(f)
    
    with open(config02, 'r') as f:
        config02 = json.load(f)
    
    with open(config03, 'r') as f:
        config03 = json.load(f)
    
    with open(config04, 'r') as f:
        config04 = json.load(f)

    with open(config05, 'r') as f:
        config05 = json.load(f)

    task_01 = predict_blockwise(**config01)
    task_02 = extract_fragments(**config02)
    task_03 = agglomerate(**config03)

    #task_02.upstream_tasks=[task_01]
    #task_03.upstream_tasks=[task_02]
    
    daisy.run_blockwise([task_01])
    daisy.run_blockwise([task_02])
    daisy.run_blockwise([task_03])

    #daisy.run_blockwise([task_01,task_02,task_03])

    find_segments(**config04)
    extract_segmentation(**config05)
