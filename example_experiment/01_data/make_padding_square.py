import numpy as np
import daisy
import sys
from multiprocessing import Pool


def change_padding_in_section(
        section_id,
        section,
        padding,
        new_dataset):

    vs = new_dataset.voxel_size

    new_section = np.pad(section,pad_width=padding)

    shape = new_section.shape

    write_roi = daisy.Roi((section_id*vs[0],0,0),(vs[0],shape[0]*vs[1],shape[1]*vs[2]))

    new_dataset[write_roi] = np.expand_dims(new_section,0)

    print(f"Written {section_id}..")


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    labels_ds = sys.argv[2]
    out_ds_name = sys.argv[3]

    try:
        output_zarr = sys.argv[4]
    except:
        print(f"No output_zarr given, using {input_zarr}")
        output_zarr = input_zarr

    labels = daisy.open_ds(input_zarr,labels_ds)
    voxel_size = labels.voxel_size
    
    array = labels.to_ndarray()
    shape = array.shape

    if ([x % 2 == 0 for x in shape[1:]] == [True,True]) or ([x % 2 == 1 for x in shape[1:]] == [True,True]):
        xy_max = max(shape[1],shape[2])
    else:
        array = np.pad(array,pad_width=((0,0),(1,0),(0,0)))
        shape = array.shape
        xy_max = max(shape[1],shape[2])

    new_shape = (shape[0],xy_max,xy_max)

    print(shape,new_shape)
    difference = np.subtract(new_shape,shape)

    padding = tuple([(int(x/2),int(x/2)) for x in difference])

    print(difference,padding)

    new_roi = daisy.Roi(
            labels.roi.offset,
            daisy.Coordinate(shape)*voxel_size + daisy.Coordinate(voxel_size[2]*difference))

    print(f"Old ROI: {labels.roi},  New ROI: {new_roi}")

    new_ds = daisy.prepare_ds(output_zarr,out_ds_name,new_roi,voxel_size,dtype=labels.dtype)

    print("Writing..")
    new_ds[new_roi] = np.pad(array,pad_width=padding) 

#    with Pool(8) as p:
#
#        p.starmap(
#                change_padding_in_section,
#                [(idx,section,padding[1:],new_ds) for idx,section in enumerate(array)]
#                )
