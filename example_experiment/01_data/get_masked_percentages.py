import zarr
import numpy as np
import daisy
import sys
from skimage.segmentation import flood_fill,flood
from multiprocessing import Pool,Manager


def get_mask_in_section(
        section_id,
        section,
        ratios):

    shape = section.shape

    mask = flood(section,(shape[0]-1,shape[1]-1))

    masked_in = np.sum(mask==False)/mask.size

    ratios.append(masked_in)
    print(f"masked_in ratio = {masked_in}..")


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]

    labels = daisy.open_ds(input_zarr,labels_ds)

    orig_array = labels.to_ndarray()

    voxel_size = labels.voxel_size
    roi = labels.roi

    manager = Manager()
    ratios = manager.list()

    with Pool(16) as p:

        p.starmap(
                get_mask_in_section,
                [(idx,section,ratios) for idx,section in enumerate(orig_array)]
                )

    print(np.mean(ratios))
