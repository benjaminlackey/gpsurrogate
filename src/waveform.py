import numpy as np
#import matplotlib.pyplot as plt
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
    """Resample using log spacing.
    """
    
    # Determine x values for resampling
    xinew = (h.x[0] if xi==None else xi)
    xfnew = (h.x[-1] if xf==None else xf)
    
    if spacing == 'linear':
        xs = np.linspace(xinew, xfnew, npoints)
    elif spacing == 'log':
        xs = np.logspace(np.log10(xinew), np.log10(xfnew), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."
    
    # Do resampling
    h.resample(xs, order=order)


def subtract_waveform_phase(h1, h2, npoints=1000, spacing='linear'):
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
    # Bounds [xi, xf] are the minimum and maximum values of x the two waveforms have in common.
    xi = max(h1.x[0], h2.x[0])
    xf = min(h1.x[-1], h2.x[-1])
    if spacing == 'linear':
        xs = np.linspace(xi, xf, npoints)
    elif spacing == 'log':
        xs = np.logspace(np.log10(xi), np.log10(xf), npoints)
    else:
        raise Exception, "Valid 'spacing' options: 'linear', 'log'."

    h1phaseint = h1.interpolate('phase')
    h2phaseint = h2.interpolate('phase')
    
    phase1 = h1phaseint(xs)
    phase2 = h2phaseint(xs)
    return Waveform({'x': xs, 'phase': phase1-phase2})


def waveform_amplitude_ratio(h1, h2, npoints=1000, spacing='linear'):
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
    # Bounds [xi, xf] are the minimum and maximum values of x the two waveforms have in common.
    xi = max(h1.x[0], h2.x[0])
    xf = min(h1.x[-1], h2.x[-1])
    if spacing == 'linear':
        xs = np.linspace(xi, xf, npoints)
    elif spacing == 'log':
        xs = np.logspace(np.log10(xi), np.log10(xf), npoints)
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



############### Functions for windowing waveforms #####################

# Add option to choose starting frequency (dx/dt/2pi) instead of starting x
# Default should be first and last data point, so no windowing is done if not specified
def window_waveform(h, xon_end, xoff_start):
    """Take two waveforms that have already been aligned in time and phase
    and smoothly transition from h1 to h2 over the window [wi, wf].
    """
    
    
    def winoff(t, wi, wf):
        return 0.5*(1.0 + np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    def winon(t, wi, wf):
        return 0.5*(1.0 - np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    # searchsorted finds index of (sorted) time where point should be inserted,
    # pushing the later points to the right.
    # indices will be just to the right of the boundaries [wi, wf]:
    ion_end = np.searchsorted(h.x, xon_end)
    ioff_start = np.searchsorted(h.x, xoff_start)
    
    # Times (window on, middle, wondow off)
    xson = h.x[:ion_end]
    xsoff = h.x[ioff_start:]
    
    hwin = h.copy()
    hwin.amp[:ion_end] *= winon(xson, hwin.x[0], xon_end)
    hwin.amp[ioff_start:] *= winoff(xsoff, xoff_start, hwin.x[-1])
    
    return hwin


def monotonic_increasing_array(y):
    """Make array y monotonic.
    Remove all elements that are less than previous largest value.
    
    Parameters
    ----------
    y : array
    
    Returns
    -------
    y_mono : array
        Subset of array that is monotonic (y_mono[i+1]>y_mono[i]).
    i_mono : arrray
        Indices for the elements in y_mono.
    """
    # If the element is not the largest, replace it with the previous largest value
    y_acc = np.maximum.accumulate(y)
    # Only keep the first unique element
    y_mono, i_mono = np.unique(y_acc, return_index=True)
    return y_mono, i_mono


def interpolate_time_of_frequency(h, order=2):
    """Generate interpolating function for t(f).
    """
    # Calculate frequency at each data point
    time = h.x
    phaseoft = h.interpolate('phase', order=order)
    omegaoft = phaseoft.derivative(n=1)
    freq = omegaoft(time)/(2*np.pi)
    
    # Delete elements that make f(t) non-monotonic.
    # then interpolate t(f)
    freq_mono, i_mono = monotonic_increasing_array(freq)
    time_mono = time[i_mono]
    
    time_of_freq = interpolate.UnivariateSpline(freq_mono, time_mono, k=order, s=0)
    
    return time_of_freq


def window_waveform_in_frequency_interval(h, fon_end, foff_start, foff_end):
    """Take two waveforms that have already been aligned in time and phase
    and smoothly transition from h1 to h2 over the window [wi, wf].
    """
    
    def winoff(t, wi, wf):
        return 0.5*(1.0 + np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    def winon(t, wi, wf):
        return 0.5*(1.0 - np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    toff = interpolate_time_of_frequency(h)
    xon_end = float(toff(fon_end))
    xoff_start = float(toff(foff_start))
    xoff_end = float(toff(foff_end))
    
    # searchsorted finds index of (sorted) time where point should be inserted,
    # pushing the later points to the right.
    # indices will be just to the right of the boundaries [wi, wf]:
    ion_end = np.searchsorted(h.x, xon_end)
    ioff_start = np.searchsorted(h.x, xoff_start)
    ioff_end = np.searchsorted(h.x, xoff_end)
    
    # Times (window on, middle, wondow off)
    xson = h.x[:ion_end]
    xsoff = h.x[ioff_start:ioff_end]
    
    hwin = h.copy()
    hwin.amp[:ion_end] *= winon(xson, hwin.x[0], xon_end)
    hwin.amp[ioff_start:ioff_end] *= winoff(xsoff, xoff_start, xoff_end)
    hwin.amp[ioff_end:] *= 0.0
    
    return hwin


########################## Fourier transform waveform ##########################

def fourier_transform(data, dt):
    """Core part of the Fourier transform.
    """
    npoints = len(data)
    data_tilde = dt*np.fft.fft(data)
    freqs = np.arange(npoints)/(npoints*dt)
    return freqs, data_tilde


def fourier_transform_waveform(h, dt):
    """Currently requires waveform to already be uniformly sampled with 
    interval dt.
    !!! This is somewhat rediculous, so allow resampling with interval dt. !!!
    """
    hresamp = h.interpolate_complex()(h.x)
    npoints = len(hresamp)
    freqs, htilde = fourier_transform(hresamp, dt)
    return Waveform.from_complex(freqs, htilde)



##################### Functions for plotting waveforms ####################

def plot_waveforms(axes, waveforms, xi=-np.inf, xf=np.inf, npoints=1000):
    axes.axhline(0.0, color='k', ls=':')
    
    for h in waveforms:
        xiplot = max(xi, h.x[0])
        xfplot = min(xf, h.x[-1])
        hcomp = h.interpolate_complex()
        times = np.linspace(xiplot, xfplot, npoints)
        hs = hcomp(times)
        
        axes.plot(times, np.real(hs))
        axes.plot(times, np.abs(hs))


def plot_waveforms_fd(ax1, ax2, waveforms, xi=None, xf=None, npoints=1000, exp=False):
    """
    """
    for h in waveforms:
        hcopy = h.copy()
        resample_uniform(hcopy, xi=xi, xf=xf, npoints=npoints, spacing='log', order=2)
        
        if exp==False:
            ax1.plot(hcopy.x, hcopy.amp)
        else:
            ax1.plot(hcopy.x, np.exp(hcopy.amp))
        ax1.set_xscale('log')
        
        ax2.plot(hcopy.x, hcopy.phase)
        ax2.set_xscale('log')

