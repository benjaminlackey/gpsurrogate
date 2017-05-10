import numpy as np
import matplotlib.pyplot as plt
import copy

import scipy.interpolate as interpolate

from constants import *


def complex_to_amp_phase(hcomplex):
    """Take a complex series and convert to amplitude and unwrapped phase.
    """
    # Calculate the amplitude
    amp = np.abs(hcomplex)

    # Calculate the angle (between -pi and pi)
    phase_wrapped = np.angle(hcomplex)

    # Unwrap the phase
    phase = np.unwrap(phase_wrapped)

    return amp, phase


class Waveform(object):
    """Generic class for either time-domain or frequency-domain waveforms.
    Can store (x, amp, phase), just (x, amp), just (x, phase), or custom set of data.
    """

    ######### Create class ##########
    def __init__(self, data):
        """Array with named columns. First column is 'x'. Or, use a dictionary to make
        a list of named 1d arrays.

        Parameters:
        ----------
        data : dictionary of equal length numpy arrays
        """
        # Make a copy of the dictionary *and* a copy of
        # its contents with deepcopy (not a reference).
        self.data = copy.deepcopy(data)

    @classmethod
    def from_amp_phase(cls, x, amp, phase, remove_start_phase=False):
        """Create Waveform from (x, amp, phase) data.
        """
        data = {'x': x, 'amp': amp, 'phase': phase}
        h = Waveform(data)
        # Zero the start phase if requested
        if remove_start_phase:
            h.add_phase(remove_start_phase=remove_start_phase)
        return h

    @classmethod
    def from_complex(cls, x, hcomplex, remove_start_phase=False):
        """Create Waveform from (x, hcomplex) data.
        """
        amp, phase = complex_to_amp_phase(hcomplex)
        return cls.from_amp_phase(x, amp, phase, remove_start_phase=remove_start_phase)

    @classmethod
    def from_hp_hc(cls, x, hplus, hcross, remove_start_phase=False):
        """Create Waveform from (x, hp+ihc) data.
        """
        hcomplex = hplus+1.0j*hcross
        return cls.from_complex(x, hcomplex, remove_start_phase=remove_start_phase)

    @classmethod
    def from_array(cls, x, y, yname='y'):
        """Create Waveform from (x, y) data.
        """
        data = {'x': x, yname: y}
        return cls(data)

    ### Getters and setters for most common arrays: x, amp, phase ###

    @property
    def x(self):
        """Can call this with h.x"""
        return self.data['x']

    @x.setter
    def x(self, xarr):
        self.data['x'] = xarr

    @property
    def amp(self):
        return self.data['amp']

    @amp.setter
    def amp(self, amparr):
        self.data['amp'] = amparr

    @property
    def phase(self):
        return self.data['phase']

    @phase.setter
    def phase(self, phasearr):
        self.data['phase'] = phasearr

    def __len__(self):
        return len(self.x)

    ########## copy Waveform ########
    def copy(self):
        """Copy the Waveform so you don't overwrite the original.
        """
        return Waveform(self.data)

    ########## Inserting data #########
    def add_data(self, y, yname='y'):
        """Add the array y to the data dictionary.
        """
        self.data[yname] = np.copy(y)

    ######## Shifting x and phase ########
    def add_x(self, x):
        """Add x to the 'x' data.
        """
        self.data['x'] += x

    def add_phase(self, phi=None, remove_start_phase=False):
        """Add phi to the phase,
        or zero the phase at the start.
        """
        # Add the phase add_phase
        if phi is not None:
            self.data['phase'] += phi
        # Shift the phase to be 0.0 at the first data point
        if remove_start_phase:
            self.data['phase'] += -self.data['phase'][0]

    ############# Interpolating data ##############
    def interpolate(self, yname, order=2):
        """Interpolate y(x) with polynomial of order order.
        """
        return interpolate.UnivariateSpline(self.data['x'], self.data[yname], k=order, s=0)

    def interpolate_complex(self, order=2, ampname='amp', phasename='phase'):
        """Interpolate complex waveform by interpolating amp(x) and phase(x).
        """
        ampoft = self.interpolate(ampname, order=order)
        phaseoft = self.interpolate(phasename, order=order)
        def comp(t):
            return ampoft(t)*np.exp(1.0j*phaseoft(t))
        return comp

    def complex(self):
        """Extract the data in complex format.
        """
        return self.amp*np.exp(1.0j*self.phase)

    ########### Resampling data ############
    def resample(self, xs, order=2):
        """Resample all the data fields at the points xs.
        """
        # Interpolate and resample each key that is not 'x'
        for key in self.data.keys():
            if key != 'x':
                key_resamp = self.interpolate(key, order=order)(xs)
                self.data[key] = key_resamp

        # Replace x with new array xs after you are done using it for interpolation
        self.data['x'] = xs


############## Resampling and comparison functions ################

def resample_uniform(h, xi=None, xf=None, npoints=1000, spacing='linear', order=2):
    """Resample waveform using uniform-linear or oniform-log spacing.
    """

    # Determine x values for resampling
    xinew = (h.x[0] if xi==None else xi)
    xfnew = (h.x[-1] if xf==None else xf)

    if spacing == 'linear':
        xs = np.linspace(xinew, xfnew, npoints)
    elif spacing == 'log':
        if xinew <= 0.0: raise Exception, "xi must be >0 if using 'log' spacing."
        xs = np.logspace(np.log10(xinew), np.log10(xfnew), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."

    # Do resampling
    h.resample(xs, order=order)


def waveform_phase_difference(h1, h2, xi=None, xf=None, npoints=1000, spacing='linear'):
    """Evaluate Phi_1(x) - Phi_2(x).

    Parameters
    ----------
    h1, h2 : Waveform
    npoints : int, optional
        Number of evenly spaced points at which to evaluate phase difference

    Returns
    -------
    Waveform object with fields 'x' and 'phase'
    """
    # Default choice of [xi, xf] are the minimum and maximum values of x
    # the two waveforms have in common.
    xinew = (max(h1.x[0], h2.x[0]) if xi==None else xi)
    xfnew = (min(h1.x[-1], h2.x[-1]) if xf==None else xf)
    if spacing == 'linear':
        xs = np.linspace(xinew, xfnew, npoints)
    elif spacing == 'log':
        if xinew <= 0.0: raise Exception, "xi must be >0 if using 'log' spacing."
        xs = np.logspace(np.log10(xinew), np.log10(xfnew), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."

    h1phaseint = h1.interpolate('phase')
    h2phaseint = h2.interpolate('phase')

    phase1 = h1phaseint(xs)
    phase2 = h2phaseint(xs)
    return Waveform({'x': xs, 'phase': phase1-phase2})


def waveform_amplitude_ratio(h1, h2, xi=None, xf=None, npoints=1000, spacing='linear'):
    """Evaluate A_1(x) / A_2(x).

    Parameters
    ----------
    h1, h2 : Waveform
    npoints : int, optional
    Number of evenly spaced points at which to evaluate phase difference

    Returns
    -------
    Waveform object with fields 'x' and 'amp'
    """
    #!!!!!!
    # You should check for zeros in the amplitude for h2.
    #!!!!!!
    # Default choice of [xi, xf] are the minimum and maximum values of x
    # the two waveforms have in common.
    xinew = (max(h1.x[0], h2.x[0]) if xi==None else xi)
    xfnew = (min(h1.x[-1], h2.x[-1]) if xf==None else xf)
    if spacing == 'linear':
        xs = np.linspace(xinew, xfnew, npoints)
    elif spacing == 'log':
        if xinew <= 0.0: raise Exception, "xi must be >0 if using 'log' spacing."
        xs = np.logspace(np.log10(xinew), np.log10(xfnew), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."

    h1ampint = h1.interpolate('amp')
    h2ampint = h2.interpolate('amp')

    amp1 = h1ampint(xs)
    amp2 = h2ampint(xs)
    return Waveform({'x': xs, 'amp': amp1/amp2})


####### Convert between physical and dimensionless units #########

def physical_to_dimensionless_time(hphys, mtot, dist):
    """Convert time-domain waveform from physical units of strain and time (s),
    to dimensionless units rescaled by the total mass and distance.

    Parameters
    ----------
    hphys : Waveform
    mtot : Total mass of binary in solar masses
    dist : Distance in Mpc

    Returns
    -------
    hdim : Waveform
        Rescaled waveform with dimensionless units
    """
    hdim = hphys.copy()
    hdim.x *= C_SI**3 / (G_SI * MSUN_SI * mtot)
    hdim.amp *= C_SI**2 * MPC_SI * dist / (G_SI * MSUN_SI * mtot)
    return hdim


def dimensionless_to_physical_time(hdim, mtot, dist):
    hphys = hdim.copy()
    hphys.x *= G_SI * MSUN_SI * mtot / C_SI**3
    hphys.amp *= G_SI * MSUN_SI * mtot / (C_SI**2 * MPC_SI * dist)
    return hphys


def physical_to_dimensionless_freq(hphys, mtot, dist):
    hdim = hphys.copy()
    hdim.x *= G_SI * MSUN_SI * mtot / C_SI**3
    hdim.amp *= C_SI**5 * MPC_SI * dist / (G_SI * MSUN_SI * mtot)**2
    return hdim


def dimensionless_to_physical_freq(hdim, mtot, dist):
    hphys = hdim.copy()
    hphys.x *= C_SI**3 / (G_SI * MSUN_SI * mtot)
    hphys.amp *= (G_SI * MSUN_SI * mtot)**2 / (C_SI**5 * MPC_SI * dist)
    return hphys


########################## Fourier transform waveform ##########################

def fourier_transform_uniform_sampled_waveform(h):
    """Fourier transform a uniformly sampled waveform.
    If possible, pad the waveform so its length is a power of 2.
    """
    dt = h.x[1]-h.x[0]
    npoints = len(h.x)

    # Put data in complex format
    data = h.amp * np.exp(1.0j*h.phase)

    # Do the Fourier transform
    data_tilde = dt*np.fft.fft(data)
    freqs = np.arange(npoints)/(npoints*dt)

    # Convert data to Waveform object
    return Waveform.from_complex(freqs, data_tilde)


##################### Functions for plotting waveforms ####################

def plot_waveforms(waveforms, xi=-np.inf, xf=np.inf, npoints=1000, amp=True, hp=True, hc=False):
    """
    """
    fig, axes = plt.subplots(1, figsize=(16, 3))

    for h in waveforms:
        xiplot = max(xi, h.x[0])
        xfplot = min(xf, h.x[-1])
        hcomp = h.interpolate_complex()
        times = np.linspace(xiplot, xfplot, npoints)
        hs = hcomp(times)
        if amp:
            axes.plot(times, np.abs(hs))
        if hp:
            axes.plot(times, np.real(hs))
        if hc:
            axes.plot(times, np.imag(hs))

    axes.axhline(0.0, color='k', ls=':')
    return fig, axes


def plot_waveforms_fd(waveforms, exp=False):
    """Plot amplitudes and phases of the waveforms.
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 6))

    for h in waveforms:
        hcopy = h.copy()
        freq = hcopy.x
        amp = hcopy.amp
        phase = hcopy.phase

        # Get rid of samples with f<=0
        posfreq = freq>0
        freq = freq[posfreq]
        amp = amp[posfreq]
        phase = phase[posfreq]

        if exp==False:
            ax1.plot(freq, amp)
        else:
            ax1.plot(freq, np.exp(amp))
        ax1.set_xscale('log')

        ax2.plot(freq, phase)
        ax2.set_xscale('log')

    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Phase')
    ax2.set_xlabel('Frequency')
    fig.subplots_adjust(hspace=0.05)

    return fig, (ax1, ax2)


def plot_waveforms_fd_resample(waveforms, xi=None, xf=None, npoints=1000, exp=False):
    """Plot amplitudes and phases of the waveforms.
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 6))

    for h in waveforms:
        hcopy = h.copy()
        if hcopy.x[0]<=0 and xi==None:
            # Don't use x=0 for log spacing
            print 'Not plotting x=0 on log plot.'
            xi=hcopy.x[1]
        resample_uniform(hcopy, xi=xi, xf=xf, npoints=npoints, spacing='log', order=2)

        if exp==False:
            ax1.plot(hcopy.x, hcopy.amp)
        else:
            ax1.plot(hcopy.x, np.exp(hcopy.amp))
        ax1.set_xscale('log')

        ax2.plot(hcopy.x, hcopy.phase)
        ax2.set_xscale('log')

    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Phase')
    ax2.set_xlabel('Frequency')
    fig.subplots_adjust(hspace=0.05)

    return fig, (ax1, ax2)


def plot_waveform_difference_fd(h1, h2, xi=None, xf=None, npoints=1000):
    """Plots of A_1/A_2 - 1 and Phi_1 - Phi_2.
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 6))
    amp_ratio = waveform_amplitude_ratio(h1, h2, xi=xi, xf=xf, npoints=npoints, spacing='log')
    phi_diff = waveform_phase_difference(h1, h2, xi=xi, xf=xf, npoints=npoints, spacing='log')

    ax1.plot(amp_ratio.x, amp_ratio.amp-1.0)
    ax1.set_xscale('log')

    ax2.plot(phi_diff.x, phi_diff.phase)
    ax2.set_xscale('log')

    ax1.axhline(0, ls=':', c='k', lw=1)
    ax2.axhline(0, ls=':', c='k', lw=1)

    ax1.set_ylabel(r'$A_1/A_2 - 1$')
    ax2.set_ylabel(r'$\Phi_1-\Phi_2$')
    ax2.set_xlabel('Frequency')
    fig.subplots_adjust(hspace=0.05)

    return fig, (ax1, ax2)
