import mahotas
import numpy as np
import daisy
import logging
import waterz
from scipy.ndimage import distance_transform_edt, measurements, label
from scipy.ndimage import gaussian_filter, maximum_filter
from funlib.segment.arrays import relabel,replace_values

logger = logging.getLogger(__name__)

def watershed_from_lsds(lsds, sigma, return_seeds=False, return_distances=False):
    '''Extract initial fragments from local shape descriptors ``lsds`` using a
    watershed transform. This assumes that the first three entries of
    ``lsds`` for each voxel are vectors pointing towards the center.'''

    boundary_distances = np.sum(lsds[0:3,:]**2, axis=0)
    boundary_distances = gaussian_filter(boundary_distances, sigma)
    boundary_distances = boundary_distances.max() - boundary_distances

    ret = watershed_from_boundary_distance(boundary_distances, return_seeds)

    if return_distances:
        ret = ret + (boundary_distances,)

    return ret

def watershed_from_affinities(
        affs,
        max_affinity_value,
        fragments_in_xy=False,
        return_seeds=False,
        min_seed_distance=10):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True'''

    if fragments_in_xy:

        mean_affs = 0.5*(affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z]>0.5*max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0)>0.5*max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            return_seeds,
            min_seed_distance=min_seed_distance)

        fragments = ret[0]

    return ret

def watershed_from_boundary_distance(
        boundary_distances,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = mahotas.label(maxima)

    logger.debug("Found %d fragments", n)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret

def watershed_in_block(
        affs,
        block,
        context,
        rag_provider,
        fragments_out,
        num_voxels_in_block,
        mask=None,
        fragments_in_xy=False,
        epsilon_agglomerate=0.0,
        filter_fragments=0.0,
        min_seed_distance=10,
        replace_sections=None):
    '''
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
        min_seed_distance (int):
            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
    '''

    total_roi = affs.roi

    logger.debug("reading affs from %s", block.read_roi)

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float32)
    else:
        max_affinity_value = 1.0

    if mask is not None:

        logger.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.debug("masking affinities")
        affs.data *= mask_data

    # extract fragments
    fragments_data, _ = watershed_from_affinities(
        affs.data,
        max_affinity_value,
        fragments_in_xy=fragments_in_xy,
        min_seed_distance=min_seed_distance)

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:

        if fragments_in_xy:
            average_affs = np.mean(affs.data[0:2]/max_affinity_value, axis=0)
        else:
            average_affs = np.mean(affs.data/max_affinity_value, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
                fragment_ids,
                measurements.mean(
                    average_affs,
                    fragments_data,
                    fragment_ids)):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(
            filtered_fragments,
            dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:

        logger.info(
            "Performing initial fragment agglomeration until %f",
            epsilon_agglomerate)

        generator = waterz.agglomerate(
                affs=affs.data/max_affinity_value,
                thresholds=[epsilon_agglomerate],
                fragments=fragments_data,
                scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                discretize_queue=256,
                return_merge_history=False,
                return_region_graph=False)
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    if replace_sections:

        logger.info("Replacing sections...")

        block_begin = block.write_roi.get_begin()
        shape = block.write_roi.get_shape()

        z_context = context[0]/affs.voxel_size[0]
        logger.info("Z context: %i",z_context)

        mapping = {}

        voxel_offset = block_begin[0]/affs.voxel_size[0]

        for i,j in zip(
                range(fragments_data.shape[0]),
                range(shape[0])):
            mapping[i] = i
            mapping[j] = int(voxel_offset + i) \
                    if block_begin[0] == total_roi.get_begin()[0] \
                    else int(voxel_offset + (i - z_context))

        logging.info('Mapping: %s', mapping)

        replace = [k for k,v in mapping.items() if v in replace_sections]

        for r in replace:
            logger.info("Replacing mapped section %i with zero", r)
            fragments_data[r] = 0

    #todo add key value replacement option

    fragments = daisy.Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1]*num_voxels_in_block
    logger.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logger.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size*daisy.Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center_z': c[0],
            'center_y': c[1],
            'center_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])
    rag.write_nodes(block.write_roi)
