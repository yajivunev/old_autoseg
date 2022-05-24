import daisy
import glob
import gunpowder as gp
import numpy as np
import os
import random
import sys
import json
import zarr


def get_sections(sample):

    csv_path = os.path.join(sample, 'csvs')

    non_empty_sections = [int(x.split('_')[-1].split('.')[0]) for x in os.listdir(csv_path)]

    return non_empty_sections

def make_raster(
        raw_file,
        raw_dataset,
        csv_file,
        read_roi):

    raw = gp.ArrayKey('RAW')
    points = gp.GraphKey('POINTS')
    raster = gp.ArrayKey('RASTER')

    source = gp.ZarrSource(
        raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=gp.Coordinate([2,2]),roi=read_roi)
            }
        )

    source = (
            gp.ZarrSource(
                filename=raw_file,
                datasets={
                    raw: raw_dataset},
                array_specs={
                    raw: gp.ArraySpec(interpolatable=True)}) +
            gp.Normalize(raw) +
            gp.Pad(raw, 0),

            gp.CsvPointsSource(
                filename=csv_file,
                points=points,
                ndims=2,
                scale=[2,2]) +
            gp.Pad(points, read_roi.shape) +
            #gp.Pad(points, labels_padding) +
            gp.RasterizeGraph(
                points,
                raster,
                array_spec=gp.ArraySpec(voxel_size=gp.Coordinate([2,2]),dtype=np.uint8,roi=read_roi),
                settings=gp.RasterizationSettings(
                    radius=(10,10),
                    mode='ball')
                )
            ) + gp.MergeProvider()


    #sources = tuple(y for x in sources for y in x)

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        #total_output_roi = total_input_roi.grow(-context, -context)
        total_output_roi = total_input_roi

    pipeline = source

#    pipeline += gp.Unsqueeze([raw])
#    pipeline += gp.Stack(1)
#
#    pipeline += gp.Squeeze([raw, raster])
#    pipeline += gp.Squeeze([raw, raster])

    raster_request = gp.BatchRequest()

    raster_request.add(raw, total_input_roi.shape)
    raster_request.add(points, total_output_roi.shape)
    raster_request.add(raster, total_output_roi.shape)

    with gp.build(pipeline):
        batch = pipeline.request_batch(raster_request)

        return batch[raster].data


if __name__ == "__main__":

    raw_file = sys.argv[1]
    raw_dataset = '2d_raw'
    out_file = raw_file
    out_dataset = 'raster'

    raw = daisy.open_ds(
            raw_file,
            'raw')
    
    voxel_size = raw.voxel_size
    shape = raw.shape[1:]

    # voxels
    input_shape_3d = gp.Coordinate((1,) + tuple(shape))
    output_shape_3d = gp.Coordinate((1,) + tuple(shape))

    # nm
    input_size_3d = input_shape_3d*voxel_size
    output_size_3d = output_shape_3d*voxel_size
    context_3d = (input_size_3d - output_size_3d) / 2

    total_roi = raw.roi.grow(-context_3d,-context_3d)

    read_roi = gp.Roi((0,0),gp.Coordinate(shape)*gp.Coordinate(voxel_size[1:]))

    out = daisy.prepare_ds(
            out_file,
            out_dataset,
            total_roi,
            voxel_size,
            dtype=np.float32)

    raster = np.zeros(shape=total_roi.shape/voxel_size)

    #get non-empty sections
    sections = get_sections(raw_file)

    for z in sections:

        print(f"at section {z}..")
        raw_ds = raw_dataset + f"/{z}"
        csv_file = os.path.join(raw_file,f'csvs/section_{z}.csv')

        raster[z] = make_raster(
                raw_file,
                raw_ds,
                csv_file,
                read_roi)

    out[total_roi] = raster
