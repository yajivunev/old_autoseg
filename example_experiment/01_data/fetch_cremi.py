import h5py
import io
import numpy as np
import requests
import zarr

if __name__ == '__main__':

    url = 'https://cremi.org/static/data/sample_B_20160501.hdf'

    in_f = h5py.File(io.BytesIO(requests.get(url, verify=False).content), 'r')

    raw = in_f['volumes/raw']
    labels = in_f['volumes/labels/neuron_ids']

    labels_mask = np.ones(shape=labels.shape).astype(np.uint8)

    out = zarr.open('cremi_sample_b.zarr', 'w')

    for ds_name, data in [
            ('raw', raw),
            ('labels', labels),
            ('labels_mask', labels_mask)]:

        print(f'writing {ds_name}')

        out[f'{ds_name}'] = data
        out[f'{ds_name}'].attrs['offset'] = [0,0,0]
        out[f'{ds_name}'].attrs['resolution'] = [40,4,4]
