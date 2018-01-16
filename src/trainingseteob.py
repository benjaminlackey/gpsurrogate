import numpy as np
import h5py

import scipy.interpolate as interpolate
import scipy.optimize as optimize
import scipy.stats as stats

import waveform as wave
import waveformset as ws
import window
import taylorf2
import trainingset as train


def time_at_max_amp(time, amp):
    """Find the time corresponding to the maximum amplitude.
    This function interpolates between data points using a 2nd-order spline,
    before finding the maximum.
    """
    nsamp = len(amp)

    # Find index of maximum amplitude
    imax = np.argmax(amp)
    # Associated time used as initial guess in numerical maximization
    tmax = time[imax]

    if imax==nsamp-1:
        raise Exception, "Maximum amplitude is at last data point. Can't calculate global maximum."

    # Range of parameters to interpolate
    # (Use no more than the 7 points surrounding the max point.)
    istart = imax-3
    iend = min(len(amp)-1, imax+4)

    # Just interpolate a small number of points surrounding imax
    tlist = time[istart:iend]
    alist = amp[istart:iend]

    # Use 2nd order interpolation
    # because it's safer and gives pretty much the same results as higher order interpolation.
    # Note: A spline is not just a local quadratic fit of 3 points.
    # It matches derivatives between local quadratics.
    # So, the answer changes as you add more points to tlist and alist.
    negampoft = interpolate.UnivariateSpline(tlist, -1.0*alist, k=2, s=0)

    # Find the minimum, starting the search at tmax
    result = optimize.minimize(negampoft, tmax)
    return result.x[0]


class ConditionedWaveform(object):
    """Methods for conditioning a time-domain Waveform object
    to be used in a frequency-domain training set.
    """

    def __init__(self, h):
        """
        """
        # Original waveform not to be manipulated
        self.h_orig = h
        # Copy of the waveform for manipulation
        self.h = h.copy()

    def resample_uniform(self, delta_t, order=2):
        """Resample waveform with spacing delta_t.
        Includes first sample, but does not necessarily include last sample.
        """
        xs = np.arange(self.h.x[0], self.h.x[-1], delta_t)
        self.h.resample(xs, order=order)

    def window_time_on(self, winon_i, winon_f, win='planck'):
        self.h = window.window_on(self.h, winon_i, winon_f, win=win)

    def window_time_off(self, winoff_i, winoff_f, win='planck'):
        self.h = window.window_off(self.h, winoff_i, winoff_f, win=win)

    def window_freq_on(self, winon_i, winon_f, win='planck'):
        self.h = window.window_freq_on(self.h, winon_i, winon_f, win=win)

    def window_freq_off(self, winoff_i, winoff_f, win='planck'):
        self.h = window.window_freq_off(self.h, winoff_i, winoff_f, win=win)

    def extend_with_zeros(self, n_ext):
        """Extend the uniformly resampled waveform with zeros so the total length is n_ext.
        The extended amplitudes will be zeros.
        The extended phases will be continuous with the last sample.

        WARNING: The waveform must be *uniformly* sampled.
        WARNING: If you are going to take a FFT, you might want n_ext to be a power of 2 for speed.
            This is the reason for the integer argument n_ext instead of the length in units of time.
        """
        # Original length
        n_orig = len(self.h.x)

        # Array of extended times
        x0 = self.h.x[0]
        delta_t = self.h.x[1] - self.h.x[0]
        times_ext = x0 + delta_t*np.arange(n_ext)

        # Extended amplitudes
        amp_ext = np.zeros(n_ext)
        amp_ext[:n_orig] = self.h.amp

        # Extended phases
        # (Make phase continuous even when amplitude goes to zero)
        phase_ext = np.ones(n_ext)*self.h.phase[-1]
        phase_ext[:n_orig] = self.h.phase

        self.h = wave.Waveform.from_amp_phase(times_ext, amp_ext, phase_ext)

    def fourier_transform(self):
        """Fourier transform the *uniformly* resampled waveform.
        This is a standard FFT tha treats the first time sample as t=0.
        You will have to do a frequency-domain time shift later if you want a
        different definition for t=0.
        """
        self.h = wave.fourier_transform_uniform_sampled_waveform(self.h)

    def zero_coalescence_time_in_frequency_domain(self, f_coalescence=None, max_amp=None):
        """Set the coalescence time to t=0, where coalescence time is determined by:
        1. Time corresponding to the frequency [f = (dphase/dt)/(2pi)] f_coalescence.
        2. Time corresponding to maximum amplitude.
        """
        t_start = self.h_orig.x[0]

        # Find coalescence time
        if (f_coalescence is not None) and (max_amp is None):
            t_coal = train.calculate_coalescence_time(self.h_orig, f_coalescence)
        elif (max_amp is not None) and (f_coalescence is None):
            t_coal = time_at_max_amp(self.h_orig.x, self.h_orig.amp)
        else:
            raise Exception, "You must pick either 'f_coalescence' or 'max_amp'."

        # Shift phase of Fourier-transformed waveform.
        dt_insp = t_coal - t_start
        self.h.phase += 2.*np.pi*self.h.x*dt_insp

    def log_spaced_in_frequency_domain(self, trunc_i, trunc_f, npoints=10000):
        wave.resample_uniform(self.h, xi=trunc_i, xf=trunc_f, npoints=npoints, spacing='log', order=2)

    # def difference_with_taylorf2(self, params):
    #     """Compare with TaylorF2 waveform.
    #     """
    #     q, spin1z, spin2z, lambda1, lambda2 = params
    #     mf = self.h.x
    #     h_ref = taylorf2.dimensionless_taylorf2_waveform(
    #         mf=mf, q=q,
    #         spin1z=spin1z, spin2z=spin2z,
    #         lambda1=lambda1, lambda2=lambda2)
    #
    #     h_ref.add_phase(remove_start_phase=True)
    #     npoints = len(mf)
    #     dh = train.lnamp_phase_difference(self.h, h_ref, npoints=npoints, spacing='log', order=2)
    #     return h_ref, dh


def moving_average_geometric_range(fs, ys, dfbyf, fbound_low=None, fbound_high=None):
    """Calculate the moving averague of ys with corresponding frequencies fs.

    Parameters
    ----------
    fs : 1d-array
        Frequencies
    ys : 1d-array
        Function values
    dfbyf : float
        Average at f is taken over the geometric range [f(1-dfbyf), f(1+dfbyf)].
        For example, dfbyf=0.1 corresponds to +/- 10% frequency interval.
    fbound_low : {float, None}
        Minimum bound for the averaging window.
    fbound_high : {float, None}
        Maximum bound for the averaging window.

    Returns
    -------
    ymovavg : 1d-array
        Moving average of ys.
    """
    if len(fs)!=len(ys):
        raise Exception, 'fs and ys must be numpy arrays with same length.'

    if fbound_low is None:
        fbound_low = fs[0]

    if fbound_high is None:
        fbound_high = fs[-1]

    ymovavg = np.zeros(len(fs))
    for i in range(len(fs)):
        f = fs[i]
        flow_try = (1.0-dfbyf)*f
        fhigh_try = (1.0+dfbyf)*f
        # Adjust interval so you don't go outside the bounds [fbound_low, fbound_high].
        flow = max(fbound_low, flow_try)
        fhigh = min(fbound_high, fhigh_try)
        # Select subset of ys samples in frequency range.
        ysubset = ys[(fs >= flow) & (fs <= fhigh)]
        # It's possible well outside [fbound_low, fbound_high] for the averaging
        # window to be empty, so don't do windowing here.
        if len(ysubset)==0:
            ysubset = ys[i]
        yavg = np.mean(ysubset)
        ymovavg[i] = yavg
    return ymovavg


def difference_with_taylorf2(h, params, quad1=None, quad2=None):
    """Compare with TaylorF2 waveform.
    """
    q, spin1z, spin2z, lambda1, lambda2 = params
    mf = h.x
    h_ref = taylorf2.dimensionless_taylorf2_waveform(
        mf=mf, q=q,
        spin1z=spin1z, spin2z=spin2z,
        lambda1=lambda1, lambda2=lambda2,
        quad1=quad1, quad2=quad2)

    h_ref.add_phase(remove_start_phase=True)
    npoints = len(mf)
    dh = train.lnamp_phase_difference(h, h_ref, npoints=npoints, spacing='log', order=2)
    return h_ref, dh


def subtract_linear_fit(dh, fi, ff):
    """Subtract a linear fit from the waveform phase.
    Line is fit in the interval (fi, ff).
    """
    # Indices for interval (fi, ff)
    fit_i = np.where((dh.x>=fi) & (dh.x<=ff))
    freq = dh.x[fit_i]
    dphase = dh.phase[fit_i]
    # Fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(freq, dphase)
    # Equation for line for all frequencies in dh
    y_fit = intercept + slope*dh.x
    # Subtract the fit
    dh.phase -= y_fit


def condition_eob_waveform(
    h, params, delta_t, n_ext,
    winon_i, winon_f,
    fit_i, fit_f,
    trunc_i, trunc_f, npoints=10000,
    win='planck',
    filter_dfbyf_amp=None, filter_dfbyf_phase=None,
    quad1=None, quad2=None,
    plots=False):
    """Generate a conditioned Frequency-domain waveform from an EOB waveform with arbitrary time samples.
    -Resample the waveform.
    -Window the beginning.
    -Pad the end with zeros so all the waveforms in the training set have the exact same time samples.
    -Fourier transform the waveform.
    -Time shift waveform so t=0 corresponds to maximum amplitude.
    -Take the residual dh with respect to TaylorF2
    -Filter dh with a moving average.
    -Resample the waveform, truncating the beginning and end.

    Parameters
    ----------
    h : Waveform
        Uniformly sampled waveform.
    winon_i : Initial frequency of on window.
    winon_f : Final frequency of on window.
    fit_i : Initial frequency for fitting \Delta\Phi to straight line.
    fit_f : Final frequency for fitting \Delta\Phi to straight line.
    trunc_i : Initial frequency of the truncated waveform after Fourier transforming.
    trunc_f : Final frequency of the truncated waveform after Fourier transforming.
    n_ext : int
        Number of samples for the extended (padded) waveform.
        All training set waveforms should have the same time samples.
        A power of 2 will make the Fourier transform efficient.
        If the phase of the Fourier-transformed waveform doesn't look correctly unwrapped,
        increase n_ext so that delta_f will be smaller.
    npoints : int
        number of logarithmically-spaced samples for the final conditioned waveform.
    win : 'hann' or 'planck'
        Type of window to use.

    Returns
    -------
    h_tilde : Waveform
        The conditioned, Fourier-transformed, and resampled waveform.
    """
    condition = ConditionedWaveform(h)

    # Resample the waveform
    condition.resample_uniform(delta_t)

    # Window the waveform
    condition.window_freq_on(winon_i, winon_f, win=win)

    if plots:
        # Windowed time-domain waveform
        fig, ax = wave.plot_waveforms([condition.h], hc=True, xi=h.x[-1]-1000, npoints=10000)
        title = '{:.3}, {:.3}, {:.3}, {:.1f}, {:.1f}'.format(params[0], params[1], params[2], params[3], params[4])
        ax.set_title(title)
        ax.minorticks_on()

    # Pad the end of the waveform with zeros
    condition.extend_with_zeros(n_ext)

    # Fourier transform
    condition.fourier_transform()

    # Resample Fourier transformed waveform with log-spacing to compress it.
    # Start at winon_i instead of 0 since you can't take the log of 0.
    htilde = condition.h
    wave.resample_uniform(htilde, xi=winon_i, npoints=npoints, spacing='log', order=2)

    # Compare with TaylorF2.
    hf2, dh = difference_with_taylorf2(htilde, params, quad1=quad1, quad2=quad2)

    # Match start with TaylorF2
    subtract_linear_fit(dh, fit_i, fit_f)

    # Filter dh with moving average.
    # Use different window widths for amplitude and phase.
    if filter_dfbyf_amp is not None:
        amp_filt = moving_average_geometric_range(
            dh.x, dh.amp, filter_dfbyf_amp,
            fbound_low=trunc_i, fbound_high=None)
    else:
        amp_filt = dh.amp
    if filter_dfbyf_phase is not None:
        phase_filt = moving_average_geometric_range(
            dh.x, dh.phase, filter_dfbyf_phase,
            fbound_low=trunc_i, fbound_high=None)
    else:
        phase_filt = dh.phase
    dh_filt = wave.Waveform.from_amp_phase(dh.x, amp_filt, phase_filt)
    # Match start with TaylorF2 *again* because the moving average filtered
    # waveform may now have a slightly better linear fit
    subtract_linear_fit(dh_filt, fit_i, fit_f)

    # Resample and truncate dh_filt then zero starting phase
    wave.resample_uniform(dh_filt, xi=trunc_i, xf=trunc_f, npoints=npoints, spacing='log', order=2)
    # Resample TaylorF2
    wave.resample_uniform(hf2, xi=trunc_i, xf=trunc_f, npoints=npoints, spacing='log', order=2)

    # Zero the start phase of the truncated waveforms
    dh_filt.add_phase(remove_start_phase=True)
    hf2.add_phase(remove_start_phase=True)

    # Reconstruct h from hf2 and the filtered dh.
    h_filt = wave.Waveform.from_amp_phase(dh_filt.x, hf2.amp*np.exp(dh_filt.amp), dh_filt.phase+hf2.phase)

    if plots:
        # Filtered residual dh of Fourier-transformed waveform relative to TaylorF2.
        fig, (ax1, ax2) = wave.plot_waveforms_fd([dh_filt])
        ax1.set_ylabel(r'$\ln(A/A_{\rm F2})$')
        ax2.set_ylabel(r'$\Phi-\Phi_{\rm F2}$')
        ax2.set_xlabel(r'$Mf$')
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.grid(which='both')
        ax2.grid(which='both')
        ax1.set_ylim(-6., 0.5)

    return h_filt, hf2, dh_filt


############### Extract training set data and condition it #####################


def get_waveform_from_training_set(f, i, mtot=2., distance=1.):
    """Get waveform in geometric units corresponding to group h_i in the hdf5 file.

    Parameters
    ----------
    f : hdf5 file handle
    i : index of waveform
    mtot : float
        Total mass (M_sun) used to generate the lalsimulation waveform.
    distance : float
        Distance (Mpc) used to generate the lalsimulation waveform.

    Returns
    -------
    params : 1d array
        Waveform parameters (smallq, s1, s2, lambda1, lambda2).
    h : Waveform
    """
    # The parameters (q, s1, s2, lambda1, lambda2)
    params = f['configurations'][i]
    # Convert bigq to smallq
    #params[0] = 1.0/params[0]

    wave_str = 'h_'+str(i)
    time = f[wave_str]['t'][:]
    amp = f[wave_str]['amp'][:]
    # The phase grows monotonically
    phase = f[wave_str]['phi'][:]

    hphys = wave.Waveform.from_amp_phase(time, amp, phase)

    # Rescale to geometric units
    h = wave.physical_to_dimensionless_time(hphys, mtot, distance)
    return params, h


# def condition_eob_training_set(
#     orig_filename, h_filename, dh_filename,
#     delta_t,
#     winon_i, winon_f,
#     n_ext,
#     trunc_i, trunc_f, npoints=10000,
#     win='planck', plots=False,
#     mtot=2.0, distance=1.0):
#     """Make a conditioned WaveformSet with waveforms from orig_filename.
#     """
#     # Open original waveform file
#     f = h5py.File(orig_filename)
#     nwave = len(f['configurations'][:])
#     print f.attrs['GenerationSettings']
#     print f['configurations_keys'][:]
#     print f['data_keys_name'][:]
#
#     # WaveformSet objects for conditioned waveforms
#     h_ts = ws.HDF5WaveformSet(h_filename)
#     dh_ts = ws.HDF5WaveformSet(dh_filename)
#
#     j = 0
#     for i in range(nwave):
#         print i, j
#         # Don't just crash if waveform is not available
#         try:
#             params, h = get_waveform_from_training_set(f, i, mtot=mtot, distance=distance)
#         except Exception as e:
#             # Exception allows for keyboard interupt
#             print e
#             print 'Not adding waveform {} to training set'.format(i)
#         else:
#             # Run if an exception was not raised
#             h_cond, hf2, dh = condition_eob_waveform(
#                 h, params, delta_t,
#                 winon_i, winon_f,
#                 n_ext,
#                 trunc_i, trunc_f, npoints=npoints,
#                 win=win, plots=plots)
#
#             h_ts.set_waveform(j, h_cond, params)
#             dh_ts.set_waveform(j, dh, params)
#             j += 1
#
#     f.close()
#     h_ts.close()
#     dh_ts.close()


def condition_eob_training_set_from_list(
    h_list, params, h_filename, dh_filename,
    delta_t, n_ext,
    winon_i, winon_f,
    fit_i, fit_f,
    trunc_i, trunc_f, npoints=10000,
    win='planck',
    filter_dfbyf_amp=None, filter_dfbyf_phase=None,
    quad1=None, quad2=None, 
    plots=False):
    """Make a conditioned WaveformSet with waveforms from h_list, params.
    """
    # WaveformSet objects for conditioned waveforms
    h_ts = ws.HDF5WaveformSet(h_filename)
    dh_ts = ws.HDF5WaveformSet(dh_filename)

    nwave = len(h_list)
    for i in range(nwave):
        print i,
        h = h_list[i]
        p = params[i]
        h_cond, hf2, dh = condition_eob_waveform(
            h, p, delta_t, n_ext,
            winon_i, winon_f,
            fit_i, fit_f,
            trunc_i, trunc_f, npoints=npoints,
            win=win,
            filter_dfbyf_amp=filter_dfbyf_amp,
            filter_dfbyf_phase=filter_dfbyf_phase,
            quad1=quad1, quad2=quad2,
            plots=plots)

        h_ts.set_waveform(i, h_cond, p)
        dh_ts.set_waveform(i, dh, p)

    h_ts.close()
    dh_ts.close()
