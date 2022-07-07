import numpy as np
from typing import List
import gunpowder as gp
import copy


def calc_max_padding(
        output_size,
        voxel_size,
        neighborhood=None,
        sigma=None,
        mode='shrink'):

    '''Calculate maximum labels padding needed.
    Args:
        output_size (array-like of ``int``):
            output size of network, in world units (a gunpowder coordinate)
        voxel_size (array-like of ``int``):
            voxel size to use (a gunpowder coordinate)
        neighborhood (``list`` of array-like, optional):
            affinity neighborhood to use.
        sigma (``int``, optional):
            sigma if using lsds
        mode (``string``, optional):
            mode to use for snapping roi to grid, see gunpowder roi
            documentation for details
    Explanation:
        when padding labels, we need to ensure that each batch still contains at
        least 50% of GT data. Additionally, we need to also consider worst case
        45 degree rotation when elastically augmenting the data. Our max padding
        is calculated as follows:
            output_size = output size of network in world coordinates (i.e \
                    nanometers not voxels)
            method_padding = largest affinity neighborhood * voxel size (for \
                    affinities) or sigma * voxel size (for lsds)
            diagonal = diagonal between x and y dimensions (i.e square root \ of
            the sum of squares of x and y axes)
            max_padding = (output_size[z]/2, diagonal/2, diagonal/2) + \
                    method_padding
        we then need to ensure max padding is a multiple of the voxel size - use
        snap_to_grid for this (see gunpowder.roi.snap_to_grid())
    '''

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
                            [np.abs(aff) for val in neighborhood \
                                    for aff in val if aff != 0]
                        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = gp.Coordinate((sigma*3,)*3)

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = gp.Roi(
                    (gp.Coordinate(
                        [i/2 for i in [output_size[0], diag, diag]]) +
                        method_padding),
                    (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()


class SwapAxes(gp.BatchFilter):
    """Swap axes of a batch given axes

    Args:
        arrays (List[ArrayKey]): ArrayKeys to swap axes for.
        axes: tuple of axes.
    """

    def __init__(self, arrays: List[gp.ArrayKey], axes: None):
        self.arrays = arrays
        self.axes = axes

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            if array in request:
                deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            if array in batch:

                outputs[array] = copy.deepcopy(batch[array])
                outputs[array].data = np.swapaxes(batch[array].data, self.axes[0],self.axes[1])
        return outputs


class BumpBackground(gp.BatchFilter):
    '''Bump background ID to max_id+1. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        label_data[label_data == 0] = np.amax(np.unique(label_data)) + 1
        batch.arrays[self.labels].data = label_data.astype(dtype)


class UnbumpBackground(gp.BatchFilter):
    '''UnBump background ID back to 0. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        uniques = np.unique(label_data)

        label_data[label_data == np.amax(uniques)] = 0

        batch.arrays[self.labels].data = label_data.astype(dtype)

