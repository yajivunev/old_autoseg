import zarr
import numpy as np
import daisy
import sys
from skimage.segmentation import flood_fill
from multiprocessing import Pool


def replace_padding_in_section(
        section_id,
        section,
        new_dataset):

    vs = new_dataset.voxel_size
    shape = section.shape
    write_roi = daisy.Roi((section_id*vs[0],0,0),(vs[0],shape[0]*vs[1],shape[1]*vs[2]))

    new_section = np.expand_dims(flood_fill(section,(shape[0]-1,shape[1]-1),0),0)

    new_dataset[write_roi] = new_section

    print(f"Written {section_id}..")


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]
    out_ds_name = sys.argv[3]

    labels = daisy.open_ds(input_zarr,labels_ds)

    orig_array = labels.to_ndarray()

    voxel_size = labels.voxel_size
    roi = labels.roi

    new_ds = daisy.prepare_ds(input_zarr,out_ds_name,roi,voxel_size,dtype=labels.dtype)

    print("Writing..")
#    new_array = np.stack([flood_fill(sxn,(0,0),0) for i,sxn in enumerate(orig_array)])

    with Pool(4) as p:

        p.starmap(
                replace_padding_in_section,
                [(idx,section,new_ds) for idx,section in enumerate(orig_array)]
                )
