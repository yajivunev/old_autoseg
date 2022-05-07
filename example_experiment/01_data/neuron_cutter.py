import zarr
import daisy
import sys
import numpy as np
import shutil

def bbox(img):

    z = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    x = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(z)[0][[0, -1]]
    cmin, cmax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(x)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def cutout_zarr(
        input_zarr,
        output_zarr,
        cutout_offset,
        cutout_shape):

    z_start = cutout_offset[0]
    y_start = cutout_offset[1]
    x_start = cutout_offset[2]
    z_shape = cutout_shape[0]
    y_shape = cutout_shape[1]
    x_shape = cutout_shape[2]

    out_zarr = zarr.open(output_zarr,"w")

    print(f"{output_zarr}: {cutout_offset},{cutout_shape}...")

    array_keys = list(input_zarr.keys())

    for key in array_keys[::-1]:

        array = input_zarr[key]

        resolution = array.attrs['resolution']
        offset = array.attrs['offset']
        
        print(f"keyK {key}, offset: {offset}{type(offset)}{offset[0]}, resolution: {resolution}")

        offset = [
                offset[0]+(z_start*resolution[0]),
                offset[1]+(y_start*resolution[1]),
                offset[2]+(x_start*resolution[2])]

        data = array[
                z_start:z_start+z_shape,
                y_start:y_start+y_shape,
                x_start:x_start+x_shape]

        print(data.shape)
            
        ds_out = out_zarr.create_dataset(
                    key,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = [0,0,0]
        ds_out.attrs['resolution'] = [50,2,2]


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    ds_name = sys.argv[2]
    label = int(sys.argv[3])
    output_dir = sys.argv[4] 

    ds = daisy.open_ds(input_zarr,ds_name)
   
    print("to ndarray'ing..")
    label_data = ds.to_ndarray()

    input_zarr = zarr.open(input_zarr,"r")

    print("bbox'ing...")
    zmin,zmax,ymin,ymax,xmin,xmax = bbox(label_data == label) 

    output_zarr = f"{output_dir}/{label}.zarr"

    cutout_offset = [zmin,ymin,xmin]
    cutout_shape = [zmax-zmin,ymax-ymin,xmax-xmin]

    print("writing..")
    cutout_zarr(input_zarr,output_zarr,cutout_offset,cutout_shape)
