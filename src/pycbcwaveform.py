import numpy as np

#import lalsimulation # Get waveform functions
#import pycbc.types # TimeSeries and FrequencySeries
import pycbc.waveform # Waveforms

from constants import *
import waveform as wave


def dimensionless_td_waveform(q, chi1, chi2, lambda1, lambda2, mf_lower, delta_tbym, approximant='TaylorT4'):
    """Wrapper for pycbc waveforms to make them dimensionless.
    Take dimensionless arguments and return dimensionless waveform.
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
                        spin1z=chi1, spin2z=chi2,
                        lambda1=lambda1, lambda2=lambda2,
                        distance=distance, inclination=inclination, 
                        delta_t=delta_t, f_lower=f_lower, f_ref=f_lower)
    
    hphys = wave.Waveform.from_hp_hc(np.array(hp.sample_times), np.array(hp), np.array(hc))
    
    # physical -> dimensionless:
    return wave.physical_to_dimensionless_time(hphys, mtot, distance)


def dimensionless_fd_waveform(q, chi1, chi2, lambda1, lambda2, mf_lower, delta_mf, approximant='TaylorF2'):
    """Wrapper for pycbc waveforms to make them dimensionless.
    Take dimensionless arguments and return dimensionless waveform.
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
    delta_f = C_SI**3 * delta_mf / (G_SI * MSUN_SI * mtot)
    
    hp, hc = pycbc.waveform.get_fd_waveform(approximant=approximant, 
                        mass1=mass1, mass2=mass2,
                        spin1z=chi1, spin2z=chi2,
                        lambda1=lambda1, lambda2=lambda2,
                        distance=distance, inclination=inclination, 
                        delta_f=delta_f, f_lower=f_lower, f_ref=f_lower)

    # Combine plus and cross polarizations: ~h(f) = ~h_+(f) + j ~h_x(f)
    hphys = wave.Waveform.from_hp_hc(np.array(hp.sample_frequencies), np.array(hp), np.array(hc))
    
    # physical -> dimensionless:
    return wave.physical_to_dimensionless_freq(hphys, mtot, distance)
