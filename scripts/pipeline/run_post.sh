python extract_fragments.py $1/02_fragments.json
python agglomerate_blockwise.py $1/03_agglomerate.json
python find_segments.py $1/04_find_segments.json
python evaluate_thresholds.py $1/evaluate.json
python extract_segmentation_from_lut.py $1/05_segmentation.json
