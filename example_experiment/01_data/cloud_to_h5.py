import numpy as np
import h5py
from cloudvolume import CloudVolume
import sys

""" Script to download a dataset through CloudVolume and write it to HDF5. """

if __name__ == '__main__':
    address = str(sys.argv[1]) #cloudvolume address
    xstart = int(sys.argv[2])
    xend = int(sys.argv[3])
    ystart = int(sys.argv[4])
    yend = int(sys.argv[5])
    zstart = int(sys.argv[6])
    zend = int(sys.argv[7])
    output_h5 = str(sys.argv[8])
    h5_dset = str(sys.argv[9])

    vol = CloudVolume(address,mip=0, use_https=True)

    print('downloading cutout..')

    cutout = vol[xstart:xend,ystart:yend,zstart:zend]
    cutout = cutout[:,:,:,0]
    cutout = np.array(cutout)
    cutout = np.swapaxes(cutout,0,2)
    #cutout = np.swapaxes(cutout,1,2)
    print('writing to h5...')

    h5file = h5py.File(output_h5,"w")
    dset = h5file.create_dataset(h5_dset,data=cutout)
    h5file.close()
