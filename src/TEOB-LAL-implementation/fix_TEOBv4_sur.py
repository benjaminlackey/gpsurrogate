import h5py
import numpy as np

fp = h5py.File('SEOBNRv4T_surrogate_v1.0.0.hdf5', 'r+')

del fp['lambda1_bounds']
del fp['lambda2_bounds']

fp.create_dataset('lambda1_bounds', data=np.array([0.0, 5000.0]))
fp.create_dataset('lambda2_bounds', data=np.array([0.0, 5000.0]))

fp.attrs['Creator'] = "Ben Lackey, Michael Puerrer, Andrea Taracchini"
fp.attrs['Email'] = "Ben.Lackey@ligo.org, Michael.Puerrer@ligo.org"
fp.attrs['Description'] = fp.attrs['description']

fp.attrs.create('version_major', 1, dtype='i4')
fp.attrs.create('version_minor', 0, dtype='i4')
fp.attrs.create('version_micro', 0, dtype='i4')

fp.close()
