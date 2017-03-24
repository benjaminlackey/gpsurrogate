import numpy as np
#import scipy.optimize as optimize

import waveform as wave
import waveformset as ws
from lalwaveform import dimensionless_td_waveform, dimensionless_fd_waveform
import window

# def truncate_beginning(h, xstart, remove_start_phase=False):
#     """Truncate Waveform below xstart and add xstart if not already a point.
#     """
#     ampint = h.interpolate('amp')
#     phaseint = h.interpolate('phase')
#     istart = np.searchsorted(h.x, xstart)
#     if xstart == h.x[istart]:
#         xnew = h.x[istart:]
#     else:
#         xnew = np.concatenate(([xstart], h.x[istart:]))
#     return wave.Waveform.from_amp_phase(xnew, ampint(xnew), phaseint(xnew), remove_start_phase=remove_start_phase)
#
#
# def condition_uniform_waveform(h, tbym_tot, mfon_end, mfoff_start, mfoff_end, mftrunc_start, win='planck'):
#     """Generate a conditioned Frequency-domain waveform from a uniformly sampled time-domain waveform.
#     1. Window the beginning and end.
#     2. Pad the end with zeros so all the waveforms in the training set have the exact same time samples.
#     3. Fourier transform the waveform.
#     4. Truncate the beginning of the Fourier transformed waveform to remove the start windowing effect.
#
#     Parameters
#     ----------
#     h : waveform
#         Uniformly sampled waveform.
#     tbym_tot : float
#         Total length of time domain waveform. Padded with zeros after merger.
#         (End time - start time)
#     mfon_end : float
#         Frequency when the on windowing stops.
#         (The on windowing starts at the first data point corresponding to mf_lower.)
#     mfoff_start : float
#         Frequency when the off windowing starts.
#     mfoff_end : float
#         Frequency when the off windowing ends.
#     mftrunc_strat : float
#         Truncate the Fourier transformed waveform below this frequency.
#         (A reasonable choice is the same as mfon_end.)
#     """
#     # Get data about the waveforms
#     n_data = len(h.x)
#     delta_tbym = h.x[1]-h.x[0]
#
#     # 1. Window the waveform
#     h = wave.window_waveform_in_frequency_interval(h, mfon_end, mfoff_start, mfoff_end, win=win)
#
#     ##### 2. Extend the waveform #####
#
#     # Set new times
#     times_ext = h.x[0] + np.arange(0.0, tbym_tot, delta_tbym)
#     n_ext = len(times_ext)
#     # Set new amplitude
#     amp_ext = np.zeros(n_ext)
#     amp_ext[:n_data] = h.amp
#     # Set new phase
#     # (Make phase continuous even when amplitude goes to zero)
#     phase_ext = np.ones(n_ext)*h.phase[-1]
#     phase_ext[:n_data] = h.phase
#
#     h_ext = wave.Waveform.from_amp_phase(times_ext, amp_ext, phase_ext)
#
#     # 3. Fourier transform
#     htilde = wave.fourier_transform_waveform(h_ext, delta_tbym)
#
#     # 4. Truncate waveform
#     htrunc = truncate_beginning(htilde, mftrunc_start, remove_start_phase=True)
#
#     # 5. TODO: You could also truncate the end if you want. Or, leave 4 and 5 outside this function.
#
#     return htrunc
#
#
# def fit_time_phase_shift(h, href, mffit_start, mffit_end, nsamp=100):
#     """Fit the difference phase-phase_ref in the interval (mffit_start, mffit_end).
#
#     Returns
#     -------
#     shifted_phase : array
#         Array to add to apply the time and phase shift in the frequency domain.
#     """
#     # Data to fit a line to
#     mfs = np.linspace(mffit_start, mffit_end, nsamp)
#     phase = h.interpolate('phase')(mfs)
#     phase_ref = href.interpolate('phase')(mfs)
#     phase_diff = phase - phase_ref
#
#     # The linear model
#     def line_fit(x, a, b):
#         return a + b*x
#
#     # Do the least squares fit
#     coeffs, covariances = optimize.curve_fit(line_fit, mfs, phase_diff)
#
#     shifted_phase = h.phase - line_fit(h.x, *coeffs)
#     hshifted = h.copy()
#     hshifted.phase = shifted_phase
#     return hshifted


################################   Utilities   #################################

def next_pow_2(time_total, delta_t):
    """Get the number of points satisfying 2**n for in n,
    that will contain a uniformly spaced array of length npoints = time_total / delta_t.

    Returns
    -------
    npow2 : int
        The next power of 2 >= npoints.
    """
    # Number of uniformly sampled points in interval T is the floor of T/dt.
    npoints = np.floor(time_total / delta_t)

    # Take base-2 log to get current power of 2,
    # Then round up and cast as int
    next_exponent = int(np.ceil(np.log2(npoints)))
    return 2**next_exponent


def zero_pn_coalescence_time(mf_low, q):
    """Time to coalescence tbym_coal starting at a frequency mf_low.
    Uses leading order (0PN) coalescence time.
    mf_low, tbym_coal are in geometric units.
    """
    eta = q/(1.0+q)**2
    return (5.0/(256.0*np.pi*mf_low*eta)) * (np.pi*mf_low)**(-5.0/3.0)


def calculate_coalescence_time(h, f_coal):
    """Calculate the time of coalescence
    using a specific frequency f_coal as the definition of coalescence time.
    """
    toff = window.interpolate_time_of_frequency(h)

    # Check that you're not requesting a frequency beyond the highest available
    fmax = toff.get_knots()[-1]
    if f_coal>fmax:
        raise Exception, 'f_coal='+str(f_coal)+' is higher than highest frequency'+str(fmax)+'.'

    t_coal = toff(f_coal)
    return float(t_coal)


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

    h1amp = h1.interpolate('amp', order=order)(xs)
    h2amp = h2.interpolate('amp', order=order)(xs)
    h1phase = h1.interpolate('phase', order=order)(xs)
    h2phase = h2.interpolate('phase', order=order)(xs)

    dlnamp = np.log(h1amp) - np.log(h2amp)
    dphase = h1phase - h2phase
    return wave.Waveform.from_amp_phase(xs, dlnamp, dphase)


def condition_waveform(h,
                       winon_i, winon_f, winoff_i, winoff_f,
                       n_ext,
                       trunc_i, trunc_f, npoints=10000,
                       win='planck', f_coalescence=None, remove_start_phase=True):
    """Generate a conditioned Frequency-domain waveform from a uniformly sampled time-domain waveform.
    1. Window the beginning and end.
    2. Pad the end with zeros so all the waveforms in the training set have the exact same time samples.
    3. Fourier transform the waveform.
    4. Optional time shift to set coalescence frequency to t=0. (Default is start of waveform at t=0.)
    5. Resample the waveform, truncating the beginning and end to remove the windowing effect.

    Parameters
    ----------
    h : Waveform
        Uniformly sampled waveform.
    winon_i : Initial frequency of on window.
    winon_f : Final frequency of on window.
    winoff_i : Initial frequency of off window.
    winoff_f : Final frequency of off window. Should be less than the ending frequency of the waveform h.
    n_ext : int
        Number of samples for the extended (padded) waveform.
        All training set waveforms should have the same time samples.
        A power of 2 will make the Fourier transform efficient.
        If the phase of the Fourier-transformed waveform doesn't look correctly unwrapped,
        increase n_ext so that delta_f will be smaller.
    trunc_i : Initial frequency of the truncated waveform after Fourier transforming.
    trunc_f : Final frequency of the truncated waveform after Fourier transforming.
    npoints : int
        number of logarithmically-spaced samples for the final conditioned waveform.
    win : 'hann' or 'planck'
        Type of window to use.
    f_coalescence : float
        Reference frequency for the Fourier transformed waveform.
        The waveform will be shifted such that the time at this frequency is at t=0.
        The shifting is done in the frequency domain.
        This is done so that the waveform is a smooth function of the waveform parameters.
    remove_start_phase : bool
        Set the phase at the start of the conditioned waveform to 0.

    Returns
    -------
    h_tilde : Waveform
        The conditioned, Fourier-transformed, and resampled waveform.
    """


    # Get data about the waveforms
    n_data = len(h.x)
    t_start = h.x[0]
    delta_t = h.x[1]-h.x[0]

    # 1. Window the waveform
    h = h.copy()
    h = window.window_freq_on(h, winon_i, winon_f, win=win)
    h = window.window_freq_off(h, winoff_i, winoff_f, win=win)

    ##### 2. Extend the waveform #####
    # Set new times
    times_ext = h.x[0] + delta_t*np.arange(n_ext)
    # Set new amplitude
    amp_ext = np.zeros(n_ext)
    amp_ext[:n_data] = h.amp
    # Set new phase
    # (Make phase continuous even when amplitude goes to zero)
    phase_ext = np.ones(n_ext)*h.phase[-1]
    phase_ext[:n_data] = h.phase
    h_ext = wave.Waveform.from_amp_phase(times_ext, amp_ext, phase_ext)

    # 3. Fourier transform
    h_tilde = wave.fourier_transform_uniform_sampled_waveform(h_ext)

    # 4. Optional time shift to set the coalescence time to t=0.
    if f_coalescence is not None:
        t_coal = calculate_coalescence_time(h, f_coalescence)
        dt_insp = t_coal - t_start
        h_tilde.phase += 2.*np.pi*h_tilde.x*dt_insp

    # 5. Resample and truncate waveform
    wave.resample_uniform(h_tilde, xi=trunc_i, xf=trunc_f, npoints=npoints, spacing='log', order=2)

    # Optionally zero the start phase of the truncated waveform
    h_tilde.add_phase(remove_start_phase=remove_start_phase)

    return h_tilde


############################ Make the training set #############################


def td_waveform_to_conditioned_fd_waveform(
    f_min, delta_t,
    winon_i, winon_f, winoff_i, winoff_f,
    n_ext,
    trunc_i, trunc_f, npoints=10000,
    win='planck', f_coalescence=None, remove_start_phase=True,
    approximant='SpinTaylorT4', amplitude_order=0,
    q=1.0,
    spin1x=0.0, spin1y=0.0, spin1z=0.0,
    spin2x=0.0, spin2y=0.0, spin2z=0.0,
    lambda1=0.0, lambda2=0.0):
    """Generate a time-domain waveform, then condition and Fourier transform it.
    """
    h = dimensionless_td_waveform(approximant=approximant, q=q,
                               spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                               spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                               lambda1=lambda1, lambda2=lambda2,
                               mf_min=f_min, delta_tbym=delta_t, amplitude_order=amplitude_order)

    h_cond = condition_waveform(h,
                       winon_i, winon_f, winoff_i, winoff_f,
                       n_ext,
                       trunc_i, trunc_f, npoints=npoints,
                       win=win, f_coalescence=f_coalescence, remove_start_phase=remove_start_phase)

    return h_cond


def td_waveform_to_conditioned_fd_waveform_difference_with_fd(
    f_min, delta_t,
    winon_i, winon_f, winoff_i, winoff_f,
    n_ext,
    trunc_i, trunc_f, ref_delta_f, npoints=10000,
    win='planck', f_coalescence=None, remove_start_phase=True,
    approximant='SpinTaylorT4', amplitude_order=0,
    q=1.0,
    spin1x=0.0, spin1y=0.0, spin1z=0.0,
    spin2x=0.0, spin2y=0.0, spin2z=0.0,
    lambda1=0.0, lambda2=0.0):
    """Generate a time-domain waveform, then condition and Fourier transform it.
    Then subtract a frequency-damain reference waveform.
    """
    h = dimensionless_td_waveform(approximant=approximant, q=q,
                spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                lambda1=lambda1, lambda2=lambda2,
                mf_min=f_min, delta_tbym=delta_t, amplitude_order=amplitude_order)

    h_cond = condition_waveform(h,
                winon_i, winon_f, winoff_i, winoff_f,
                n_ext,
                trunc_i, trunc_f, npoints=npoints,
                win=win, f_coalescence=f_coalescence, remove_start_phase=remove_start_phase)

    # Generate FD wavefrom between trunc_i and trunc_f with same parameters
    h_ref = dimensionless_fd_waveform(approximant='TaylorF2',
                q=q,
                spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                lambda1=lambda1, lambda2=lambda2,
                mf_min=trunc_i, mf_max=trunc_f, delta_mf=ref_delta_f)

    wave.resample_uniform(h_ref, npoints=npoints, spacing='log', order=2)

    h_ref.add_phase(remove_start_phase=remove_start_phase)

    dh = lnamp_phase_difference(h_cond, h_ref, npoints=npoints, spacing='log', order=2)
    return h_cond, dh


def make_training_set(
    h_filename, dh_filename, params,
    f_min, delta_t,
    winon_i, winon_f, winoff_i, winoff_f,
    n_ext,
    trunc_i, trunc_f, ref_delta_f, npoints=10000,
    win='planck', f_coalescence=None, remove_start_phase=True,
    approximant='SpinTaylorT4', amplitude_order=0):
    """Make a WaveformSet containing waveforms with parameters params.
    """
    h_ts = ws.HDF5WaveformSet(h_filename)
    dh_ts = ws.HDF5WaveformSet(dh_filename)
    for i in range(len(params)):
        p = params[i]
        q, spin1z, spin2z, lambda1, lambda2 = p
        print i, 

        h, dh = td_waveform_to_conditioned_fd_waveform_difference_with_fd(
            f_min, delta_t,
            winon_i, winon_f, winoff_i, winoff_f,
            n_ext,
            trunc_i, trunc_f, ref_delta_f, npoints=npoints,
            win=win, f_coalescence=f_coalescence, remove_start_phase=remove_start_phase,
            approximant=approximant, amplitude_order=amplitude_order,
            q=q,
            spin1z=spin1z, spin2z=spin2z,
            lambda1=lambda1, lambda2=lambda2)

        h_ts.set_waveform(i, h, p)
        dh_ts.set_waveform(i, dh, p)

    h_ts.close()
    dh_ts.close()






# # kwargs is a dictionary ({key : value} pairs)
# # **kwargs unpacks this dictionary and returns keyword arguments to a function
# def get_td_vs_fd_waveform(mf_min, delta_tbym, tbym_tot, mfon_end, mfoff_start, mfoff_end,
#                                     mftrunc_start, href_mf_max, href_delta_mf, mffit_start, mffit_end, ndownsample=1000,
#                                     htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2', win='planck', **kwargs):
#     """Generate a time-domain waveform and compare it to a frequency-domain waveform.
#
#
#     Returns
#     -------
#     hfshifted : Waveform
#         TD waveform after Conditioning, Fourier transforming, and matching at beginning to FD waveform.
#     hfref : Waveform
#         FD reference waveform
#     dhf : Waveform
#         lnA_TD(Mf)-lnA_FD(Mf) and Phi_TD(Mf)-Phi_FD(Mf) stored in the 'amp' and 'phase' arrays.
#     """
#     ########### Calculate time-domain waveform and condition it in the frequency domain ############
#     ht = dimensionless_td_waveform(mf_min=mf_min, delta_tbym=delta_tbym, approximant=htapproximant,
#                                    amplitude_order=-1, **kwargs)
#     hfcond = condition_uniform_waveform(ht, tbym_tot, mfon_end, mfoff_start, mfoff_end, mftrunc_start, win=win)
#     wave.resample_uniform(hfcond, npoints=ndownsample, spacing='log', order=2)
#
#     ########### Calculate reference frequency domain waveform #############
#     # Zero the waveform parameters that aren't allowed in a specific waveform approximant
#     if hfrefapproximant=='IMRPhenomP':
#         kwargs['lambda1'] = kwargs['lambda2'] = 0.0
#     if hfrefapproximant=='TaylorF2':
#         kwargs['spin1x'] = kwargs['spin1y'] = kwargs['spin2x'] = kwargs['spin2y'] = 0.0
#     hfref = dimensionless_fd_waveform(mf_min=mf_min, mf_max=href_mf_max, delta_mf=href_delta_mf, approximant=hfrefapproximant, **kwargs)
#     wave.resample_uniform(hfref, npoints=ndownsample, spacing='log', order=2)
#
#     # Shift the waveform to match the reference waveform
#     hfshifted = fit_time_phase_shift(hfcond, hfref, mffit_start, mffit_end, nsamp=1000)
#
#     # Calculate the lnamp and phase differences
#     dhf = lnamp_phase_difference(hfshifted, hfref, npoints=ndownsample, spacing='log', order=2)
#
#     # Make 3 separate WaveformSet objects that contain lists of hfref, hfshifted, dh
#     #return ht, hfcond, hfshifted, hfref, dhf
#     return hfshifted, hfref, dhf
#
#
# def make_td_vs_fd_waveform_set(filename, params,
#                                mf_lower, delta_tbym, tbym_tot,
#                                mfon_end, mfoff_start, mfoff_end,
#                                mftrunc_start,
#                                href_mf_max, href_delta_mf,
#                                mffit_start, mffit_end,
#                                htapproximant='SpinTaylorT4', hfrefapproximant='TaylorF2', win='planck'):
#     """Make a WaveformSet containing waveforms with parameters params.
#     """
#     ts = ws.HDF5WaveformSet(filename)
#
#     for i in range(len(params)):
#         print i,
#         p = params[i]
#         q, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, lambda1, lambda2 = p
#         hfshifted, hfref, dhf = get_td_vs_fd_waveform(mf_lower, delta_tbym, tbym_tot,
#                                     mfon_end, mfoff_start, mfoff_end,
#                                     mftrunc_start,
#                                     href_mf_max, href_delta_mf,
#                                     mffit_start, mffit_end,
#                                     ndownsample=1000,
#                                     htapproximant=htapproximant, hfrefapproximant=hfrefapproximant,
#                                     win=win,
#                                     q=q,
#                                     spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
#                                     spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
#                                     lambda1=lambda1, lambda2=lambda2)
#         ts.set_waveform(i, dhf, p)
#
#     return ts
