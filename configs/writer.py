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
        "num_cache_workers": 4,
        }
extract_frags = {
        "base_dir": base_dir,
        "epsilon_agglomerate": 0.0,
        "num_workers": 20,
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
        "num_workers": 20,
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
        "num_workers": 20,
        "fragments_file": "{}/{}/01_data/{}/{}/{}.zarr",
        "block_size": [0, 0, 0],
        "thresholds_minmax": [0, 1],
        "edges_collection": "edges_hist_quant_75"
        }

extract_seg = {
        "num_workers": 20,
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

pipeline = {
        "affs_file":"",
        "affs_dataset":"",
        "fragments_dataset":"",
        "fragments_in_xy": True,
        "epsilon_agglomerate":0.0,
        "filter_fragments":0.05,
        "replace_sections":None,
        "merge_function":"hist_quant_75",
        "gt_file":None,
        "gt_dataset":None,
        "mask_file":None,
        "mask_dataset":None,
        "roi_offset":None,
        "roi_shape":None,
        "run_type":None
        }
        

def config_writer(experiment, setups, vol):

    inf_path = '../scripts/pipeline/jobs/{}_inference_{}'.format(vol,experiment)
    postproc_path = '../scripts/pipeline/jobs/{}_postproc_{}'.format(vol,experiment)
    pipeline_path = '../scripts/pipeline/jobs/{}_pipeline_{}'.format(vol,experiment)
    inf_script = open(inf_path,"a")
    pproc_script = open(postproc_path,"a")
    pipeline_script = open(pipeline_path,"a")

    for setup in setups.keys():
        for iteration in setups[setup]:
            
            #hard-code because lazy
            
            fragments_file = "{}/{}/01_data/{}/{}/{}.zarr".format(base_dir,experiment,setup,iteration,vol)
            file_name = "{}.zarr".format(vol)
            raw_dataset = "clahe_raw" if experiment=="glia" else "raw"
            block_size = [15000,1250,1250] if experiment!="cremi" else [12000,2500,2500]
            context = [1000,40,40] if experiment!="cremi" else [800,80,80]
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

            if setup.endswith("tf"):
                predict["num_workers"] = 3
                predict["num_cache_workers"] = 10

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
    
            gt_file =  os.path.join(base_dir,experiment,"01_data",file_name)
            gt_dataset = "glia" if experiment=="glia" else "labels"
            
            voi["gt_file"] = gt_file
            voi["gt_dataset"] = gt_dataset
            voi["fragments_file"] = fragments_file
            voi["roi_offset"] = roi_offset
            voi["roi_shape"] = roi_shape
            voi["results_file"] = os.path.join(fragments_file,"results.out")

            extract_seg["block_size"] = block_size
            extract_seg["fragments_file"] = fragments_file
            extract_seg["threshold"] = threshold
            extract_seg["out_dataset"] = f"segmentation_{threshold}"

            pipeline["affs_file"] = fragments_file
            pipeline["affs_dataset"] = "affs"
            pipeline["fragments_dataset"] = "fragments"
            pipeline["gt_file"] = gt_file
            pipeline["gt_dataset"] = gt_dataset
            pipeline["roi_offset"] = roi_offset
            pipeline["roi_shape"] = roi_shape

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

            pipeline_config = os.path.join(output_dir,"pipeline_{}.json".format(vol))
            with open(pipeline_config,'w') as f:
                json.dump(pipeline,f)

            predict_cmd = "python ../predict_blockwise.py {}".format(os.path.abspath(predict_config))
            frags_cmd = "python ../extract_fragments.py {}".format(os.path.abspath(extract_frags_config))
            agglomerate_cmd = "python ../agglomerate_blockwise.py {}".format(os.path.abspath(agglomerate_config))
            find_segs_cmd = "python ../find_segments.py {}".format(os.path.abspath(find_segments_config))
            segs_cmd = "python ../extract_segmentation_from_lut.py {}".format(os.path.abspath(extract_segments_config))
            voi_cmd = "python ../evaluate_thresholds.py {}".format(os.path.abspath(voi_config))
    
            pipeline_cmd = "python ../pipeline.py {}".format(os.path.abspath(pipeline_config))

            inf_script.write("{}\n".format(predict_cmd))
            pproc_script.write("{}; {}; {}; {}; {}\n".format(frags_cmd,agglomerate_cmd,find_segs_cmd,voi_cmd,segs_cmd))
            pipeline_script.write(pipeline_cmd+"\n")

    inf_script.close()
    pproc_script.close()
    pipeline_script.close()
    subprocess.call(['chmod','+x',inf_path])
    subprocess.call(['chmod','+x',postproc_path])
    subprocess.call(['chmod','+x',pipeline_path])

if __name__ == "__main__":

    #experiment = "cremi"
    #experiment = "neuron"
    experiment = "glia"
    setups = {
            "affs_pt": [100000],
            "mtlsd_0_pt": [150000],
            "mtlsd_1_pt": [51000],
            "mtlsd_2_pt": [100000],
            "mtlsd_3_pt": [115000],
            "affs_tf": [100000],
            "mtlsd_0_tf": [150000],
            "mtlsd_1_tf": [51000],
            "mtlsd_2_tf": [100000],
            "mtlsd_3_tf": [115000]
            #"vanilla_dense_tf":[60000],
            #"mtlsd_dense_tf":[49000],
            #"vanilla_sparse_tf":[60000],
            #"vanilla_dense_pt":[60000],
            #"mtlsd_dense_pt":[49000],
            #"vanilla_sparse_pt":[60000],
            #"vanilla_dense_pt":[100000],
            #"mtlsd_dense_pt":[80000],
            #"vanilla_sparse_pt":[80000],
            #"mtlsd_sparse_pt": [60000],
            #"vanilla_dense_tf":[100000],
            #"mtlsd_dense_tf":[80000],
            #"vanilla_sparse_tf":[80000],
            #"mtlsd_sparse_tf": [60000]
            }
    vol = "oblique"
    #vol = "cremi_sample_c"
    config_writer(experiment,setups,vol)
