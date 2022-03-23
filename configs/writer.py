import json
import sys
import os
import subprocess

""" Script to write all config files for the inference pipeline 
(predict_blockwise, extract_fragments, agglomerate, find_segments,
extract_segmentation_from_lut, evaluate_voi.
"""

base_dir = "/scratch1/04101/vvenu/autoseg"

predict = {
        "base_dir": base_dir,
        "raw_dataset": "", 
        "experiment": "", 
        "num_workers": 4,
        "file_name": "{}.zarr",
        "setup": "",
        "iteration": 0,
        "num_cache_workers": 10,
        }
extract_frags = {
        "base_dir": base_dir,
        "epsilon_agglomerate": 0.0,
        "num_workers": 32,
        "fragments_in_xy": True,
        "block_size": [0,0,0],
        "setup": "{}",
        "fragments_dataset": "fragments",
        "iteration": 0,
        "experiment": "",
        "filter_fragments": 0.05,
        "context": [0, 0, 0],
        "affs_dataset": "affs",
        "file_name": ""
        }
agglomerate = {
        "base_dir": base_dir,
        "num_workers": 32,
        "merge_function": "hist_quant_75",
        "block_size": [0, 0, 0],
        "setup": "{}",
        "fragments_dataset": "fragments",
        "iteration": 0,
        "experiment": "",
        "context": [0, 0, 0],
        "affs_dataset": "affs",
        "file_name": ""
        }
find_segments = {
        "thresholds_step": 0.02,
        "fragments_dataset": "fragments",
        "num_workers": 32,
        "fragments_file": "{}/{}/01_data/{}/{}/{}.zarr",
        "block_size": [0, 0, 0],
        "thresholds_minmax": [0, 1],
        "edges_collection": "edges_hist_quant_75"
        }

extract_seg = {
        "num_workers": 32,
        "block_size": [0,0,0],
        "fragments_dataset": "fragments",
        "fragment_file":"",
        "threshold":0.48,
        "edges_collection": "edges_hist_quant_75",
        "out_dataset": ""
        }

voi = {
        "fragments_dataset": "fragments",
        "roi_offset": None,
        "gt_file": "{}/{}/01_data/{}/{}.zarr",
        "roi_shape": None,
        "gt_dataset": "{}",
        "thresholds_step": 0.02,
        "fragments_file": "{}/{}/01_data/{}/{}/{}.zarr",
        "thresholds_minmax": [0, 1],
        "edges_collection": "edges_hist_quant_75"
        }

def config_writer(experiment, setups, vol):

    inf = '../scripts/pipeline/{}_inference_jobs_{}'.format(vol,experiment)
    postproc = '../scripts/pipeline/{}_postproc_jobs_{}'.format(vol,experiment)
    inf_script = open(inf,"a")
    pproc_script = open(postproc,"a")

    for setup in setups.keys():
        for iteration in setups[setup]:
            fragments_file = "{}/{}/01_data/{}/{}/{}.zarr".format(base_dir,experiment,setup,iteration,vol)
            file_name = "{}.zarr".format(vol)
            raw_dataset = "clahe_raw" if experiment=="glia" else "raw"
            block_size = [300,300,300] if experiment!="cremi" else [400,400,400]
            context = [50,50,50] if experiment!="cremi" else [40,40,40]
            threshold = 0.98 if experiment=="glia" else 0.48

            if vol == "oblique":
                roi_offset = [1000,1800,460]
                roi_shape = [500,1200,2660]
            elif vol == "cremi_sample_c":
                roi_offset = [600,0,0]
                roi_shape = [2000,5000,5000]
            else:
                roi_offset = None
                roi_shape = None

            #modifying dictionaries

            predict["experiment"] = experiment
            predict["setup"] = setup
            predict["iteration"] = int(iteration)
            predict["raw_dataset"] = raw_dataset
            predict["file_name"] = file_name

            extract_frags["experiment"] = experiment
            extract_frags["file_name"] = file_name
            extract_frags["setup"] = setup
            extract_frags["iteration"] = int(iteration)
            extract_frags["block_size"] = block_size
            extract_frags["context"] = context

            agglomerate["setup"] = setup
            agglomerate["iteration"] = int(iteration)
            agglomerate["block_size"] = block_size
            agglomerate["file_name"] = file_name
            agglomerate["experiment"] = experiment
            agglomerate["context"] = context

            find_segments["fragments_file"] = fragments_file
            find_segments["block_size"] = block_size
    
            voi["gt_file"] = os.path.join(base_dir,experiment,"01_data",file_name)
            voi["gt_dataset"] = "glia" if experiment=="glia" else "labels"
            voi["fragments_file"] = fragments_file
            voi["roi_offset"] = roi_offset
            voi["roi_shape"] = roi_shape
            voi["results_file"] = os.path.join(fragments_file,"results.out")

            extract_seg["block_size"] = block_size
            extract_seg["fragments_file"] = fragments_file
            extract_seg["threshold"] = threshold
            extract_seg["out_dataset"] = f"segmentation_{threshold}"

            output_dir = os.path.join('.config', experiment, setup, str(iteration))

            try:
                os.makedirs(output_dir)
            except:
                pass

            predict_config = os.path.join(output_dir,"predict_{}.json".format(vol))
            with open(predict_config, 'w') as f:
                json.dump(predict,f)

            extract_frags_config = os.path.join(output_dir,"extract_frags_{}.json".format(vol))
            with open(extract_frags_config, 'w') as f:
                json.dump(extract_frags,f)

            agglomerate_config = os.path.join(output_dir,"agglomerate_{}.json".format(vol))
            with open(agglomerate_config, 'w') as f:
                json.dump(agglomerate,f)

            find_segments_config = os.path.join(output_dir,"find_segments_{}.json".format(vol))
            with open(find_segments_config, 'w') as f:
                json.dump(find_segments,f)

            extract_segments_config = os.path.join(output_dir,"extract_segments_{}.json".format(vol))
            with open(extract_segments_config, 'w') as f:
                json.dump(extract_seg,f)
            
            voi_config = os.path.join(output_dir,"voi_{}.json".format(vol))
            with open(voi_config, 'w') as f:
                json.dump(voi,f)

            predict_cmd = "python predict_blockwise.py {}".format(os.path.abspath(predict_config))
            frags_cmd = "python extract_fragments.py {}".format(os.path.abspath(extract_frags_config))
            agglomerate_cmd = "python agglomerate_blockwise.py {}".format(os.path.abspath(agglomerate_config))
            find_segs_cmd = "python find_segments.py {}".format(os.path.abspath(find_segments_config))
            segs_cmd = "python extract_segments.py {}".format(os.path.abspath(extract_segments_config))
            voi_cmd = "python ../evaluate_thresholds.py {}".format(os.path.abspath(voi_config))
            
            inf_script.write("{}\n".format(predict_cmd))
            pproc_script.write("{}; {}; {}; {}; {}\n".format(frags_cmd,agglomerate_cmd,find_segs_cmd,segs_cmd,voi_cmd))

    inf_script.close()
    pproc_script.close()
    subprocess.call(['chmod','+x',inf])
    subprocess.call(['chmod','+x',postproc])

if __name__ == "__main__":

    experiment = "cremi"
    setups = {
            "mtlsd_dense_pt":[49000],
            #"mtlsd_dense_tf":[49000],
            "vanilla_dense_pt":[60000],
            #"vanilla_dense_tf":[60000],
            "vanilla_sparse_pt":[60000],
            #"vanilla_sparse_tf":[60000]
            }

    vol = "cremi_sample_c"
    config_writer(experiment,setups,vol)
