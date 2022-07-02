import copy
from typing import List
import gunpowder as gp
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
import numpy as np

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


class BoostWeights(gp.BatchFilter):
    ''' Boost weights wherever booster != 0.'''

    def __init__(self,booster,weights,amount):
        self.booster = booster
        self.weights = weights
        self.amount = amount

    def process(self, batch, request):
        booster_data = batch.arrays[self.booster].data
        weights_data = batch.arrays[self.weights].data
        dtype = weights_data.dtype

        batch.arrays[self.weights].data = (weights_data + self.amount*(booster_data.astype(bool)).astype(dtype))


class BumpBackground(gp.BatchFilter):
    '''Bump background ID to max_id+1. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        label_data[label_data == 0] = np.amax(np.unique(label_data)) + 2
        batch.arrays[self.labels].data = label_data.astype(dtype)

class UnbumpBackground(gp.BatchFilter):
    '''UnBump background ID back to 0. '''

    def __init__(self,labels):
        self.labels = labels

    def process(self, batch, request):
        label_data = batch.arrays[self.labels].data
        dtype = label_data.dtype

        uniques = np.unique(label_data)

        if len(uniques) == 1:
            label_data[label_data == np.amax(uniques)] = 0

        elif len(uniques) == 2:
            label_data[label_data == np.amax(uniques)] = 0
            label_data[label_data == np.amin(uniques)] = 1

        else: raise AssertionError("more than 2 actual classes!")

        batch.arrays[self.labels].data = label_data.astype(dtype)

class BinaryDilation(gp.BatchFilter):
    '''Find connected components of the same value, and replace each component
    with a new label.

    Args:

        labels (:class:`ArrayKey`):

            The label array to modify.
    '''

    def __init__(self, labels,iterations):
        self.labels = labels
        self.iterations = iterations

    def process(self, batch, request):
        components = batch.arrays[self.labels].data
        dtype = components.dtype
        #simple_neighborhood = malis.mknhood3d()
        #affinities_from_components = malis.seg_to_affgraph(
        #    components,
        #    simple_neighborhood)
        #components, _ = malis.connected_components_affgraph(
        #    affinities_from_components,
        #    simple_neighborhood)
        components = binary_dilation(components,iterations=self.iterations)
        batch.arrays[self.labels].data = components.astype(dtype)

class ComputeDT(gp.BatchFilter):

    def __init__(
            self,
            labels,
            sdt,
            constant=0.5,
            dtype=np.float32,
            mode='3d',
            dilate_iterations=None,
            scale=None,
            mask=None,
            labels_mask=None,
            unlabelled=None):

        self.labels = labels
        self.sdt = sdt
        self.constant = constant
        self.dtype = dtype
        self.mode = mode
        self.dilate_iterations = dilate_iterations
        self.scale = scale
        self.mask = mask
        self.labels_mask = labels_mask
        self.unlabelled = unlabelled

    def setup(self):

        spec = self.spec[self.labels].copy()

        self.provides(self.sdt,spec)

        if self.mask:
            self.provides(self.mask, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.labels] = request[self.sdt].copy()

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()

        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()

        return deps

    def _compute_dt(self, data):

        dist_func = distance_transform_edt

        if self.dilate_iterations:
            data = binary_dilation(
                    data,
                    iterations=self.dilate_iterations)

        if self.scale:
            inner = dist_func(binary_erosion(data))
            outer = dist_func(np.logical_not(data))

            distance = (inner - outer) + self.constant

            distance = np.tanh(distance / self.scale)

        else:

            inner = dist_func(data) - self.constant
            outer = -(dist_func(1-np.logical_not(data)) - self.constant)

            distance = np.where(data, inner, outer)

        return distance.astype(self.dtype)

    def process(self, batch, request):

        outputs = gp.Batch()

        labels_data = batch[self.labels].data
        distance = np.zeros_like(labels_data).astype(self.dtype)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.sdt].roi.copy()
        spec.dtype = np.float32

        labels_data = labels_data != 0

        # don't need to compute on entirely background batches
        if np.sum(labels_data) != 0:

            if self.mode == '3d':
                distance = self._compute_dt(labels_data)

            elif self.mode == '2d':
                for z in range(labels_data.shape[0]):
                    distance[z] = self._compute_dt(labels_data[z])
            else:
                raise ValueError('Only implemented for 2d or 3d labels')
                return

        if self.mask and self.mask in request:

            if self.labels_mask:
                mask = batch[self.labels_mask].data
            else:
                mask = (labels_data!=0).astype(self.dtype)

            if self.unlabelled:
                unlabelled_mask = batch[self.unlabelled].data
                mask *= unlabelled_mask

            outputs[self.mask] = gp.Array(
                    mask.astype(self.dtype),
                    spec)

        outputs[self.sdt] =  gp.Array(distance, spec)

        return outputs
