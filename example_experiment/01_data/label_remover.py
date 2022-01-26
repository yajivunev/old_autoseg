import sys
import zarr
import numpy as np

""" Script to remove labels provided in a labeled dataset. """

def in1d_alternative_2D(npArr, arr):
    idx = np.searchsorted(arr, npArr.ravel())
    idx[idx==len(arr)] = 0
    return arr[idx].reshape(npArr.shape) == npArr

if __name__ == '__main__':
    input_zarr = str(sys.argv[1]) #path to zarr directory
    labels_ds = str(sys.argv[2]) #labels dataset name
    n=len(sys.argv[3]) #third arg should be list of labels to be removed without spaces
    a=sys.argv[3][1:n-1]
    labels_list=a.split(',')
    labels_list = list(map(int, labels_list))

    input_zarr = zarr.open(input_zarr,"r+")

    print("reading input zarr datasets...")

    labels = input_zarr[labels_ds]
    
    resolution = labels.attrs['resolution']    
    offset = labels.attrs['offset']

    print("Unlabelling given labels...")

    unlabelled = np.array(labels)
    labels_list = np.array(labels_list)

    unlabelled[in1d_alternative_2D(unlabelled,labels_list)] = 0

    #for label in labels_list:
    #    unlabelled[unlabelled == label] = 0

    print("writing to zarr...")

    for ds_name, data in [
            (labels_ds+'_cleaned', unlabelled)]:

        ds_out = input_zarr.create_dataset(
                    ds_name,
                    data=data,
                    compressor=zarr.get_codec(
                        {'id': 'gzip', 'level': 5}
                    ))

        ds_out.attrs['offset'] = offset
        ds_out.attrs['resolution'] = resolution

