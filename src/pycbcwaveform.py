import numpy as np

import pycbc.waveform

from constants import *
import waveform as wave


def dimensionless_td_waveform(q=1.0, 
                              spin1x=0.0, spin1y=0.0, spin1z=0.0, 
                              spin2x=0.0, spin2y=0.0, spin2z=0.0, 
                              lambda1=0.0, lambda2=0.0,
                              amplitude_order=-1, phase_order=-1, 
                              mf_lower=0.001, delta_tbym=10.0, approximant='SpinTaylorT4'):
    """Wrapper for pycbc waveforms to make them dimensionless.
    Take dimensionless arguments and return dimensionless waveform.
    Spins are defined at the reference frequency which is set to mf_lower here.
    
    Parameters
    ----------
    q : The small mass ratio q=m2/m1<=1 where m1 >= m2.
    spin1x, spin1y, spin1z, lambda1 : Spins and tidal parameter of the more massive star m1.
    spin2x, spin2y, spin2z, lambda2 : Spins and tidal parameter of the less massive star m2.
    
    Returns
    -------
    h : Waveform
    """
    # dimensionless -> physical:
    # Pick fiducial mtot = 1Msun and distance = 1Mpc
    # pycbc expects units of Msun and Mpc
    mtot = 1.0
    distance = 1.0
    inclination = 0.0
    
    mass1 = mtot / (1.0 + q)
    mass2 = mtot * q / (1.0 + q)
    
    f_lower = C_SI**3 * mf_lower / (G_SI * MSUN_SI * mtot)
    delta_t = G_SI * MSUN_SI * mtot * delta_tbym / C_SI**3
    
    hp, hc = pycbc.waveform.get_td_waveform(approximant=approximant, 
                        mass1=mass1, mass2=mass2,
                        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, 
                        lambda1=lambda1, lambda2=lambda2, 
                        distance=distance, inclination=inclination,
                        amplitude_order=amplitude_order, phase_order=phase_order, 
                        delta_t=delta_t, f_lower=f_lower, f_ref=f_lower)
    
    # Zero the start time and phase.
    hphys = wave.Waveform.from_hp_hc(np.array(hp.sample_times), np.array(hp), np.array(hc))
    hphys.add_x(-hphys.x[0])
    hphys.add_phase(remove_start_phase=True)
    
    # physical -> dimensionless:
    return wave.physical_to_dimensionless_time(hphys, mtot, distance)


def dimensionless_fd_waveform(q=1.0, 
                              spin1x=0.0, spin1y=0.0, spin1z=0.0, 
                              spin2x=0.0, spin2y=0.0, spin2z=0.0, 
                              lambda1=0.0, lambda2=0.0,
                              amplitude_order=-1, phase_order=-1, 
                              mf_lower=0.001, delta_mf=1.0e-6, approximant='TaylorF2'):
    """Wrapper for pycbc waveforms to make them dimensionless.
    Take dimensionless arguments and return dimensionless waveform.
    Spins are defined at the reference frequency which is set to mf_lower here.
    
    Parameters
    ----------
    q : The small mass ratio q=m2/m1<=1 where m1 >= m2.
    spin1x, spin1y, spin1z, lambda1 : Spins and tidal parameter of the more massive star m1.
    spin2x, spin2y, spin2z, lambda2 : Spins and tidal parameter of the less massive star m2.
    
    Returns
    -------
    h : Waveform
    """
    # dimensionless -> physical:
    # Pick fiducial mtot = 1Msun and distance = 1Mpc
    # pycbc expects units of Msun and Mpc
    mtot = 1.0
    distance = 1.0
    inclination = 0.0
    
    # Remove sections of the waveform that have an amplitude below this threshold (e.g. beginning and end).
    ampthresh = 1.0e-12
    
    mass1 = mtot / (1.0 + q)
    mass2 = mtot * q / (1.0 + q)
    
    f_lower = C_SI**3 * mf_lower / (G_SI * MSUN_SI * mtot)
    delta_f = C_SI**3 * delta_mf / (G_SI * MSUN_SI * mtot)
    
    hp, hc = pycbc.waveform.get_fd_waveform(approximant=approximant, 
                        mass1=mass1, mass2=mass2,
                        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, 
                        lambda1=lambda1, lambda2=lambda2,
                        distance=distance, inclination=inclination, 
                        amplitude_order=amplitude_order, phase_order=phase_order, 
                        delta_f=delta_f, f_lower=f_lower, f_ref=f_lower)

    # Combine plus and cross polarizations: ~h(f) = ~h_+(f) + j ~h_x(f)
    hphys = wave.Waveform.from_hp_hc(np.array(hp.sample_frequencies), np.array(hp), np.array(hc))
    
    # physical -> dimensionless:
    hdim =  wave.physical_to_dimensionless_freq(hphys, mtot, distance)
    
    # Remove the start of the waveform that is just zeros
    #iabove = np.searchsorted(hdim.x, mf_lower)
#     iabove = np.searchsorted(hdim.amp, ampthresh)
#     return wave.Waveform.from_amp_phase(hdim.x[iabove:], hdim.amp[iabove:], hdim.phase[iabove:])
    i_nonzero = np.where(hdim.amp>ampthresh)
    return wave.Waveform.from_amp_phase(hdim.x[i_nonzero], hdim.amp[i_nonzero], hdim.phase[i_nonzero])

