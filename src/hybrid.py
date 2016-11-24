import numpy as np

#import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.optimize as optimize

from waveform import *

def convolution(ampoft1, phaseoft1, ampoft2, phaseoft2, ti, tf, tau=0.0):
    """Calculate the convolution z(tau) = \int_ti^tf h1(t) h2^*(t-\tau) dt.
    It has the real and imaginary parts z(tau) = x(tau) + y(tau).
    """
    def real_part(t):
        return ampoft1(t)*ampoft2(t-tau)*(np.cos(phaseoft1(t)-phaseoft2(t-tau)))
    
    def imag_part(t):
        return ampoft1(t)*ampoft2(t-tau)*(np.sin(phaseoft1(t)-phaseoft2(t-tau)))
    
    x = integrate.quad(real_part, ti, tf)[0]
    y = integrate.quad(imag_part, ti, tf)[0]
    return x + 1.0j*y


def sigma(ampoft, ti, tf, tau=0.0):
    """Normalization constant for waveform in interval [ti, tf].
    (Optionally time shift the signal from t -> t-tau,
    to calculate normalization constant between ti-tau and tf-tau.)
    """
    # tprime = t - tau
    # dtprime = dt
    tprimei = ti - tau
    tprimef = tf - tau
    return np.sqrt(integrate.quad(ampoft, tprimei, tprimef)[0])


def overlap(ampoft1, phaseoft1, ampoft2, phaseoft2, ti, tf, tau=0.0):
    """The complex overlap between h1 and h2 in the time interval [ti, tf].
    h2 is shifted from t -> t-tau
    """
    z = convolution(ampoft1, phaseoft1, ampoft2, phaseoft2, ti, tf, tau)
    sigma1 = sigma(ampoft1, ti, tf)
    sigma2 = sigma(ampoft2, ti, tf, tau)
    
    return z/(sigma1*sigma2)


def calculate_time_phase_shift(h1, h2, t1i, t1f, taua, taub):
    """Find the time and phase shift for waveform 2 that maximizes the overlap between h1 and h2
    in the interval [t1i, t1f] for waveform 1.
    
    h1 stays fixed.
    h2 is shifted.
    """
    # Put error checking here:
    # Calculate minimum and maximum allowed time shifts.
    amp1int = h1.interpolate('amp')
    phase1int = h1.interpolate('phase')
    amp2int = h2.interpolate('amp')
    phase2int = h2.interpolate('phase')
    
    # Function to minimize
    def neg_overlap_amp(tau):
        """The function you are minimizing to find the best time shift.
        """
        return -np.abs(overlap(amp1int, phase1int, amp2int, phase2int, t1i, t1f, tau))
    
    # Search for minimum of neg_overlap_amp between taua and taub
    bounds = (taua, taub)
    overlapbest = optimize.minimize_scalar(neg_overlap_amp, bounds=bounds, method='bounded')
    
    # Time and phase shift needed for h2
    taubest = overlapbest.x
    comp_overlap_best = overlap(amp1int, phase1int, amp2int, phase2int, t1i, t1f, taubest)
    phibest = -np.angle(comp_overlap_best)
    
    return taubest, phibest


def construct_hybrid(h1, h2, wi, wf):
    """Take two waveforms that have already been aligned in time and phase
    and smoothly transition from h1 to h2 over the window [wi, wf].
    """
    # find times closest to boundaries w1, wf
    # convert data to complex
    # do window
    # convert back to complex
    
    def winoff(t):
        return 0.5*(1.0 + np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    def winon(t):
        return 0.5*(1.0 - np.cos( (np.pi*(t-wi))/(wf-wi) ) )
    
    # searchsorted finds index of (sorted) time where point should be inserted,
    # pushing the later points to the right.
    # indices will be just to the right of the boundaries [wi, wf]:
    i1winstart = np.searchsorted(h1.data['x'], wi)
    i1winend = np.searchsorted(h1.data['x'], wf)
    i2winend = np.searchsorted(h2.data['x'], wf)
    
    # Times (before, in, after) hybridization window
    times1 = h1.data['x'][:i1winstart]
    times1hyb = h1.data['x'][i1winstart:i1winend]
    times2 = h2.data['x'][i2winend:]
    
    # Construct interpolating functions for the complex waveforms
    h1compint = h1.interpolate_complex()
    h2compint = h2.interpolate_complex()
    
    # Interpolate at the desired times, and do the windowing
    h1comp = h1compint(times1)
    hhybcomp = winoff(times1hyb)*h1compint(times1hyb)+winon(times1hyb)*h2compint(times1hyb)
    h2comp = h2compint(times2)
    
    # Join (before, during, after) the windowing
    times = np.concatenate((times1, times1hyb, times2))
    hcomp = np.concatenate((h1comp, hhybcomp, h2comp))
    
    # hybridized waveform
    return Waveform.from_hp_hc(times, np.real(hcomp), np.imag(hcomp))


def construct_inspiral_nr_hybrid(hinsp, hnr, tnrmatchi, tnrmatchf, tnrwini, tnrwinf):
    """
    Parameters
    ----------
    tnrmatchi : float
        Initial time of matching window from the maximum amplitude of the NR waveform.
    tnrmatchf : float
        Final time of matching window from the maximum amplitude of the NR waveform.
    """
    # What you should really do is start by shifting
    # both inspiral and NR to have t=0 at max amplitude (merger).
    # ...
    
    # Shift NR waveform time to be zero at max amplitude
    hnrshift = hnr.copy()
    nrmaxi = np.argmax(hnrshift.data['amp'])
    tnrmax = hnrshift.data['x'][nrmaxi]
    hnrshift.add_x(-tnrmax)
    
#     fig, axes = plt.subplots(1, figsize=(16, 6))
#     hplots = WaveformPlot(hinsp, hnrshift)
#     hplots.plot_waveforms(axes, -4000, 4000)
#     axes.minorticks_on()
    
    # Maximum and minimum times to search for best match
    shift_insp_min = tnrmatchf - hinsp.data['x'][-1]
    shift_insp_max = -tnrmatchi*2.0
    
#     print tnrmatchi, tnrmatchf
#     print shift_insp_min, shift_insp_max
    
    # Find time to shift inspiral waveform to match fixed NR waveform 
    taubest, phibest = calculate_time_phase_shift(hnrshift, hinsp, tnrmatchi, tnrmatchf, shift_insp_min, shift_insp_max)
    #print taubest, phibest
    
    # Now shift NR waveform (again) in the opposite direction
    hnrshift.add_x(-taubest)
    hnrshift.add_phase(phibest)
    
    # Make Construct the hybrid
    hhybrid = construct_hybrid(hinsp, hnrshift, tnrwini-taubest, tnrwinf-taubest)
    
    return hhybrid, hnrshift, taubest