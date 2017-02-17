import numpy as np
import scipy.optimize as optimize

import waveform as wave
import waveformset as ws
#from pycbcwaveform import dimensionless_td_waveform, dimensionless_fd_waveform
from lalwaveform import dimensionless_td_waveform, dimensionless_fd_waveform
import taylorf2 as f2

def truncate_beginning(h, xstart, remove_start_phase=False):
    """Truncate Waveform below xstart and add xstart if not already a point.
    """
    ampint = h.interpolate('amp')
    phaseint = h.interpolate('phase')
    istart = np.searchsorted(h.x, xstart)
    if xstart == h.x[istart]:
        xnew = h.x[istart:]
    else:
        xnew = np.concatenate(([xstart], h.x[istart:]))
    return wave.Waveform.from_amp_phase(xnew, ampint(xnew), phaseint(xnew), remove_start_phase=remove_start_phase)


def condition_uniform_waveform(h, tbym_tot, mfon_end, mfoff_start, mfoff_end, mftrunc_start, win='planck'):
    """Generate a conditioned Frequency-domain waveform from a uniformly sampled time-domain waveform. 
    1. Window the beginning and end.
    2. Pad the end with zeros so all the waveforms in the training set have the exact same time samples.
    3. Fourier transform the waveform.
    4. Truncate the beginning of the Fourier transformed waveform to remove the start windowing effect.
    
    Parameters
    ----------
    h : waveform
        Uniformly sampled waveform.
    tbym_tot : float
        Total length of time domain waveform. Padded with zeros after merger.
        (End time - start time)
    mfon_end : float
        Frequency when the on windowing stops. 
        (The on windowing starts at the first data point corresponding to mf_lower.)
    mfoff_start : float
        Frequency when the off windowing starts.
    mfoff_end : float
        Frequency when the off windowing ends.
    mftrunc_strat : float
        Truncate the Fourier transformed waveform below this frequency.
        (A reasonable choice is the same as mfon_end.)
    """    
    # Get data about the waveforms
    n_data = len(h.x)
    delta_tbym = h.x[1]-h.x[0]
    
    # 1. Window the waveform
    h = wave.window_waveform_in_frequency_interval(h, mfon_end, mfoff_start, mfoff_end, win=win)
    
    ##### 2. Extend the waveform #####
    
    # Set new times
    times_ext = h.x[0] + np.arange(0.0, tbym_tot, delta_tbym)
    n_ext = len(times_ext)
    # Set new amplitude
    amp_ext = np.zeros(n_ext)
    amp_ext[:n_data] = h.amp
    # Set new phase
    # (Make phase continuous even when amplitude goes to zero)
    phase_ext = np.ones(n_ext)*h.phase[-1]
    phase_ext[:n_data] = h.phase
    
    h_ext = wave.Waveform.from_amp_phase(times_ext, amp_ext, phase_ext)
    
    # 3. Fourier transform
    htilde = wave.fourier_transform_waveform(h_ext, delta_tbym)
    
    # 4. Truncate waveform
    htrunc = truncate_beginning(htilde, mftrunc_start, remove_start_phase=True)
    
    # 5. TODO: You could also truncate the end if you want. Or, leave 4 and 5 outside this function.
    
    return htrunc


def fit_time_phase_shift(h, href, mffit_start, mffit_end, nsamp=100):
    """Fit the difference phase-phase_ref in the interval (mffit_start, mffit_end).
    
    Returns
    -------
    shifted_phase : array
        Array to add to apply the time and phase shift in the frequency domain.
    """
    # Data to fit a line to
    mfs = np.linspace(mffit_start, mffit_end, nsamp)
    phase = h.interpolate('phase')(mfs)
    phase_ref = href.interpolate('phase')(mfs)
    phase_diff = phase - phase_ref
        
    # The linear model
    def line_fit(x, a, b):
        return a + b*x
    
    # Do the least squares fit
    coeffs, covariances = optimize.curve_fit(line_fit, mfs, phase_diff)
    
    shifted_phase = h.phase - line_fit(h.x, *coeffs)
    hshifted = h.copy()
    hshifted.phase = shifted_phase
    return hshifted


def lnamp_phase_difference(h1, h2, npoints=1000, spacing='linear', order=2):
    """Evaluate lnA_1(x)-lnA_2(x) and Phi_1(x)-Phi_2(x).
    
    Parameters
    ----------
    h1, h2 : Waveform
    npoints : int, optional
        Number of evenly spaced points at which to evaluate phase difference
    
    Returns
    -------
    Waveform
    """
    # Bounds [xi, xf] are the minimum and maximum values of x the two waveforms have in common.
    xi = max(h1.x[0], h2.x[0])
    xf = min(h1.x[-1], h2.x[-1])
    if spacing == 'linear':
        xs = np.linspace(xi, xf, npoints)
    elif spacing == 'log':
        xs = np.logspace(np.log10(xi), np.log10(xf), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."

    h1lnamp = h1.interpolate('amp', order=order)(xs)
    h2lnamp = h2.interpolate('amp', order=order)(xs)
    h1phase = h1.interpolate('phase', order=order)(xs)
    h2phase = h2.interpolate('phase', order=order)(xs)
    
    dlnamp = np.log(h1lnamp) - np.log(h2lnamp)
    dphase = h1phase - h2phase
    return wave.Waveform.from_amp_phase(xs, dlnamp, dphase)


############################ Make the training set #############################


# kwargs is a dictionary ({key : value} pairs)
# **kwargs unpacks this dictionary and returns keyword arguments to a function
def get_td_vs_fd_waveform(mf_min, delta_tbym, tbym_tot, mfon_end, mfoff_start, mfoff_end,
                                    mftrunc_start, href_mf_max, href_delta_mf, mffit_start, mffit_end, ndownsample=1000,
                                    htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2', win='planck', **kwargs):
    """Generate a time-domain waveform and compare it to a frequency-domain waveform.
    
    
    Returns
    -------
    hfshifted : Waveform
        TD waveform after Conditioning, Fourier transforming, and matching at beginning to FD waveform.
    hfref : Waveform
        FD reference waveform
    dhf : Waveform
        lnA_TD(Mf)-lnA_FD(Mf) and Phi_TD(Mf)-Phi_FD(Mf) stored in the 'amp' and 'phase' arrays.
    """
    ########### Calculate time-domain waveform and condition it in the frequency domain ############
    ht = dimensionless_td_waveform(mf_min=mf_min, delta_tbym=delta_tbym, approximant=htapproximant,
                                   amplitude_order=-1, **kwargs)
    hfcond = condition_uniform_waveform(ht, tbym_tot, mfon_end, mfoff_start, mfoff_end, mftrunc_start, win=win)
    wave.resample_uniform(hfcond, npoints=ndownsample, spacing='log', order=2)
    
    ########### Calculate reference frequency domain waveform #############
    # Zero the waveform parameters that aren't allowed in a specific waveform approximant
    if hfrefapproximant=='IMRPhenomP':
        kwargs['lambda1'] = kwargs['lambda2'] = 0.0
    if hfrefapproximant=='TaylorF2':
        kwargs['spin1x'] = kwargs['spin1y'] = kwargs['spin2x'] = kwargs['spin2y'] = 0.0
    hfref = dimensionless_fd_waveform(mf_min=mf_min, mf_max=href_mf_max, delta_mf=href_delta_mf, approximant=hfrefapproximant, **kwargs)
    wave.resample_uniform(hfref, npoints=ndownsample, spacing='log', order=2)
    
    # Shift the waveform to match the reference waveform
    hfshifted = fit_time_phase_shift(hfcond, hfref, mffit_start, mffit_end, nsamp=1000)
    
    # Calculate the lnamp and phase differences
    dhf = lnamp_phase_difference(hfshifted, hfref, npoints=ndownsample, spacing='log', order=2)
    
    # Make 3 separate WaveformSet objects that contain lists of hfref, hfshifted, dh
    #return ht, hfcond, hfshifted, hfref, dhf
    return hfshifted, hfref, dhf


def make_td_vs_fd_waveform_set(filename, params, 
                               mf_lower, delta_tbym, tbym_tot, 
                               mfon_end, mfoff_start, mfoff_end, 
                               mftrunc_start, 
                               href_mf_max, href_delta_mf,
                               mffit_start, mffit_end,
                               htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2', win='planck'):
    """Make a WaveformSet containing waveforms with parameters params.
    """
    ts = ws.HDF5WaveformSet(filename)
    
    for i in range(len(params)):
        print i,
        p = params[i]
        q, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, lambda1, lambda2 = p
        hfshifted, hfref, dhf = get_td_vs_fd_waveform(mf_lower, delta_tbym, tbym_tot, 
                                    mfon_end, mfoff_start, mfoff_end, 
                                    mftrunc_start, 
                                    href_mf_max, href_delta_mf,
                                    mffit_start, mffit_end, 
                                    ndownsample=1000,
                                    htapproximant=htapproximant, hfrefapproximant=hfrefapproximant,
                                    win=win,
                                    q=q, 
                                    spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
                                    spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, 
                                    lambda1=lambda1, lambda2=lambda2)
        ts.set_waveform(i, dhf, p)

    return ts





## kwargs is a dictionary ({key : value} pairs)
## **kwargs unpacks this dictionary and returns keyword arguments to a function
#def get_td_vs_fd_waveform(mf_lower, delta_tbym, tbym_tot, mfon_end, mfoff_start, mfoff_end, 
#                                    mftrunc_start, href_delta_mf, mffit_start, mffit_end, ndownsample=1000, 
#                                    htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2', **kwargs):
#    """Generate a time-domain waveform and compare it to a frequency-domain waveform.
#    
#    
#    Returns
#    -------
#    hfshifted : Waveform
#        TD waveform after Conditioning, Fourier transforming, and matching at beginning to FD waveform.
#    hfref : Waveform
#        FD reference waveform
#    dhf : Waveform
#        lnA_TD(Mf)-lnA_FD(Mf) and Phi_TD(Mf)-Phi_FD(Mf) stored in the 'amp' and 'phase' arrays.
#    """
#    ########### Calculate time-domain waveform and condition it in the frequency domain ############
##     ht = dimensionless_td_waveform(mf_lower=mf_lower, delta_tbym=delta_tbym, approximant=htapproximant, **kwargs)
#    ht = dimensionless_td_waveform(mf_lower=mf_lower, delta_tbym=delta_tbym, approximant=htapproximant, 
#                                   amplitude_order=-1, **kwargs)
#    hfcond = condition_uniform_waveform(ht, tbym_tot, mfon_end, mfoff_start, mfoff_end, mftrunc_start)
#    wave.resample_uniform(hfcond, npoints=ndownsample, spacing='log', order=2)
#    
#    ########### Calculate reference frequency domain waveform #############
#    # Zero the waveform parameters that aren't allowed in a specific waveform approximant
#    if hfrefapproximant=='IMRPhenomP':
#        kwargs['lambda1'] = kwargs['lambda2'] = 0.0
#    if hfrefapproximant=='TaylorF2':
#        kwargs['spin1x'] = kwargs['spin1y'] = kwargs['spin2x'] = kwargs['spin2y'] = 0.0
#    hfref = dimensionless_fd_waveform(mf_lower=mf_lower, delta_mf=href_delta_mf, approximant=hfrefapproximant, **kwargs)
#    wave.resample_uniform(hfref, npoints=ndownsample, spacing='log', order=2)
#    
#    # Shift the waveform to match the reference waveform
#    hfshifted = fit_time_phase_shift(hfcond, hfref, mffit_start, mffit_end, nsamp=1000)
#    
#    # Calculate the lnamp and phase differences
#    dhf = lnamp_phase_difference(hfshifted, hfref, npoints=ndownsample, spacing='log', order=2)
#    
#    # Make 3 separate WaveformSet objects that contain lists of hfref, hfshifted, dh
#    #return ht, hfcond, hfshifted, hfref, dhf
#    return hfshifted, hfref, dhf
#
#
#def make_td_vs_fd_waveform_set(filename, params, 
#                               mf_lower, delta_tbym, tbym_tot, 
#                               mfon_end, mfoff_start, mfoff_end, 
#                               mftrunc_start, 
#                               href_delta_mf, 
#                               mffit_start, mffit_end,
#                               htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2'):
#    """Make a WaveformSet containing waveforms with parameters params.
#    """
#    ts = ws.HDF5WaveformSet(filename)
#    
#    for i in range(len(params)):
#        print i,
#        p = params[i]
#        q, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, lambda1, lambda2 = p
#        hfshifted, hfref, dhf = get_td_vs_fd_waveform(mf_lower, delta_tbym, tbym_tot, 
#                                    mfon_end, mfoff_start, mfoff_end, 
#                                    mftrunc_start, 
#                                    href_delta_mf, 
#                                    mffit_start, mffit_end, 
#                                    ndownsample=1000,
#                                    htapproximant=htapproximant, hfrefapproximant=hfrefapproximant,
#                                    q=q, 
#                                    spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, 
#                                    spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, 
#                                    lambda1=lambda1, lambda2=lambda2)
#        ts.set_waveform(i, dhf, p)
#
#    return ts
#
