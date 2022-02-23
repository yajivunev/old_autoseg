import sys
import daisy
import skimage.measure
from funlib.segment.arrays import relabel_connected_components


if __name__ == "__main__":

    input_zarr = sys.argv[1]
    in_ds = sys.argv[2]
    out_ds = sys.argv[3]

    ds_in = daisy.open_ds(input_zarr,in_ds)

    voxel_size = ds_in.voxel_size
    roi = ds_in.roi
    chunk_shape = ds_in.chunk_shape
    write_size = chunk_shape*voxel_size

    ds_out = daisy.prepare_ds(
            input_zarr,
            out_ds,
            roi,
            voxel_size,
            dtype=ds_in.dtype,
            write_size=write_size)

    relabel_connected_components(ds_in,ds_out,write_size,10)
