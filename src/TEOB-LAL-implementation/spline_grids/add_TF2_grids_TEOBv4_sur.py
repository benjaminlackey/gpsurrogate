import h5py
import numpy as np

fp = h5py.File('SEOBNRv4T_surrogate_v1.0.0.hdf5', 'r+')

D = '/Users/mpuer/Documents/gpsurrogate/src/TEOB-LAL-implementation/spline_grids/'
Mf_amp_cubic = np.load(D+'Mf_amp_TF2_cubic.npy')
Mf_phi_cubic = np.load(D+'Mf_phi_TF2_cubic.npy')
Mf_amp_linear = np.load(D+'Mf_amp_TF2_linear.npy')
Mf_phi_linear = np.load(D+'Mf_phi_TF2_linear.npy')

try:
    del fp['TF2_Mf_amp']
    del fp['TF2_Mf_phi']
except Exception:
    pass
try:
    del fp['TF2_Mf_amp_cubic']
    del fp['TF2_Mf_phi_cubic']
except Exception:
    pass
try:
    del fp['TF2_Mf_amp_linear']
    del fp['TF2_Mf_phi_linear']
except Exception:
    pass

fp.create_dataset('TF2_Mf_amp_cubic', data=Mf_amp_cubic)
fp.create_dataset('TF2_Mf_phi_cubic', data=Mf_phi_cubic)
fp.create_dataset('TF2_Mf_amp_linear', data=Mf_amp_linear)
fp.create_dataset('TF2_Mf_phi_linear', data=Mf_phi_linear)

fp.close()
