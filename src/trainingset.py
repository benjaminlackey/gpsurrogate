import numpy as np
import scipy.optimize as optimize

import waveform as wave
import waveformset as ws
from pycbcwaveform import dimensionless_td_waveform
import taylorf2 as f2

def truncate_beginning(h, xstart, remove_start_phase=False):
    """
    """
    ampint = h.interpolate('amp')
    phaseint = h.interpolate('phase')
    istart = np.searchsorted(h.x, xstart)
    if xstart == h.x[istart]:
        xnew = h.x[istart:]
    else:
        xnew = np.concatenate(([xstart], h.x[istart:]))
    return wave.Waveform.from_amp_phase(xnew, ampint(xnew), phaseint(xnew), remove_start_phase=remove_start_phase)


def conditioned_waveform(q, chi1, chi2, lambda1, lambda2, mf_lower,
                         delta_tbym, length, mfon_end, mfoff_start, mfoff_end, mftrunc_start, 
                         approximant='SpinTaylorT4'):
    """
    """
    # Generate waveform
    h = dimensionless_td_waveform(q, chi1, chi2, lambda1, lambda2, mf_lower, delta_tbym, approximant=approximant)
    
    # Window the waveform
    h = wave.window_waveform_in_frequency_interval(h, mfon_end, mfoff_start, mfoff_end)
    
    # extend the waveform
    ndata = len(h.x)
    times = h.x[0] + np.arange(0.0, length, delta_tbym)
    ntimes = len(times)
    ampall = np.zeros(ntimes)
    # Make phase continuous even when amplitude goes to zero (for fun)...
    phaseall = np.ones(ntimes)*h.phase[-1]
    ampall[:ndata] = h.amp
    phaseall[:ndata] = h.phase
    hextend = wave.Waveform.from_amp_phase(times, ampall, phaseall)
    
    #fig, axes = plt.subplots(1, figsize=(16, 6))
    #plot_waveforms(axes, [h], xi=-np.inf, xf=np.inf, npoints=1000)
    
    # Fourier transform
    htilde = wave.fourier_transform_waveform(hextend, delta_tbym)
    
    # Truncate waveform
    htrunc = truncate_beginning(htilde, mftrunc_start, remove_start_phase=True)
    
#     # Use the same time shift for all waveforms
#     htrunc.phase -= 2.0*np.pi*htrunc.x*tshift
#     htrunc.add_phase(remove_start_phase=True)
    return htrunc


def fit_time_phase_shift(freq, phase, phase_anal, f_fitend):
    """Fit the difference phase-phase_anal in the interval [freq[0], f_fitend).
    
    """
    # Data to fit a line to
    diff = phase - phase_anal
    i_fitend = np.searchsorted(freq, f_fitend)
    freq_data = freq[:i_fitend]
    diff_data = diff[:i_fitend]
        
    # The linear model
    def line_fit(x, a, b):
        return a + b*x
    
    # Do the least squares fit
    coeffs, covariances = optimize.curve_fit(line_fit, freq_data, diff_data)
    
    shifted_phase = phase - line_fit(freq, *coeffs)
    return shifted_phase


def match_taylorf2_at_beginning(h, q, chi1, chi2, lambda1, lambda2, f_fitend):
    """
    """
    # Taylor f2 phase
    f2_phase = f2.taylorf2_phase(h.x, 0.0, 0.0, q/(1+q)**2, chi1, chi2, lambda1, lambda2)
    
    # Calculated the Shifted phase that matches TaylorF2 at the beginning
    shifted_phase = fit_time_phase_shift(h.x, h.phase, f2_phase, f_fitend)

    # Wrap then unwrap phase to correct for insufficient sampling of original waveform h
    comp_data = h.amp*np.exp(1.0j*shifted_phase)
    hnew = wave.Waveform.from_complex(h.x, comp_data)
    
    # Change the start phase to exactly that of TaylorF2
    hnew.add_phase(f2_phase[0]-hnew.phase[0])
    
    return hnew

############################ Make the training set #############################

def make_training_set(filename, points, mf_lower, delta_tbym, length, mfon_end,
                      mfoff_start, mfoff_end, mftrunc_start, f_fitend):
    """Make a waveform set.
    """
    ts = ws.HDF5WaveformSet(filename)
    
    for i in range(len(points)):
        p = points[i]
        print i,
        #print(i, end="\r")
        h = conditioned_waveform(p[0], p[1], p[2], p[3], p[4], mf_lower, delta_tbym,
                                 length, mfon_end, mfoff_start, mfoff_end, mftrunc_start)
        h = match_taylorf2_at_beginning(h, p[0], p[1], p[2], p[3], p[4], f_fitend)
        ts.set_waveform(i, h, p)

    return ts

