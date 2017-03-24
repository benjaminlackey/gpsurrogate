import numpy as np
import scipy.interpolate as interpolate

def hann_on(x, xi, xf):
    """Hann on window function. Only works with floats for now.
    """
    if x<=xi:
        return 0.0
    elif x>=xf:
        return 1.0
    else:
        return 0.5*(1.0 - np.cos( (np.pi*(x-xi))/(xf-xi) ) )


def hann_off(x, xi, xf):
    if x<=xi:
        return 1.0
    elif x>=xf:
        return 0.0
    else:
        return 0.5*(1.0 + np.cos( (np.pi*(x-xi))/(xf-xi) ) )


def planck(x):
    """Might want to deal with overflow and underflow of exp(x)
    with an expansion, but this works fine.
    """
    if x < -100.0:
        return 1.0
    elif x > 100.0:
        return 0.0
    else:
        return 1.0/(np.exp(x)+1.0)


def planck_on(x, xi, xf):
    if x<=xi:
        return 0.0
    elif x>=xf:
        return 1.0
    else:
        z = (xf-xi)/(x-xi) + (xf-xi)/(x-xf)
        return planck(z)


def planck_off(x, xi, xf):
    if x<=xi:
        return 1.0
    elif x>=xf:
        return 0.0
    else:
        # z has a minus sign relative to planck_on
        z = (xi-xf)/(x-xi) + (xi-xf)/(x-xf)
        return planck(z)


def monotonic_increasing_array(y):
    """Make array y monotonic by
    removing all elements that are less than the previous largest value.
    Useful for dealing with numerical errors that cause a slowly increasing
    function to be slightly non-monotonic.

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

    # Delete elements that make f(t) non-monotonic (due to numerical error),
    # then interpolate t(f)
    freq_mono, i_mono = monotonic_increasing_array(freq)
    time_mono = time[i_mono]

    time_of_freq = interpolate.UnivariateSpline(freq_mono, time_mono, k=order, s=0)
    return time_of_freq


def window_on(h, xi, xf, win='planck'):
    """Smoothly turn on the waveform amplitude from 0 to A(x) over the window [xi, xf].
    """
    if win=='hann':
        w_on = np.array([hann_on(x, xi, xf) for x in h.x])
    elif win=='planck':
        w_on = np.array([planck_on(x, xi, xf) for x in h.x])
    else:
        raise Exception, "only 'hann' and 'planck' windows are implemented."
    h.amp *= w_on
    return h


def window_off(h, xi, xf, win='planck'):
    """Smoothly turn off the waveform amplitude from A(x) to 0 over the window [xi, xf].
    """
    if win=='hann':
        w_off = np.array([hann_off(x, xi, xf) for x in h.x])
    elif win=='planck':
        w_off = np.array([planck_off(x, xi, xf) for x in h.x])
    else:
        raise Exception, "only 'hann' and 'planck' windows are implemented."
    h.amp *= w_off
    return h


def window_freq_on(h, freqi, freqf, win='planck', order=2, eps=1.0e-4):
    """Smoothly turn on the waveform amplitude from 0 to A(x) over the frequency window [freqi, freqf].
    eps :
        Fractional tolerance when determining if [freqi, freqf] are within
        the range of frequencies of h.
    """
    toff = interpolate_time_of_frequency(h, order=order)
    fmin = toff.get_knots()[0]
    fmax = toff.get_knots()[-1]
    if freqi<(1.0-eps)*fmin:
        raise Exception, 'freqi='+str(freqi)+' is lower than lowest frequency '+str(fmin)+'.'
    if freqf>(1.0+eps)*fmax:
        raise Exception, 'freqi='+str(freqf)+' is higher than highest frequency '+str(fmax)+'.'

    xi = float(toff(freqi))
    xf = float(toff(freqf))
    return window_on(h, xi, xf, win=win)


def window_freq_off(h, freqi, freqf, win='planck', order=2, eps=1.0e-4):
    """Smoothly turn off the waveform amplitude from A(x) to 0 over the frequency window [freqi, freqf].
    eps :
        Fractional tolerance when determining if [freqi, freqf] are within
        the range of frequencies of h.
    """
    toff = interpolate_time_of_frequency(h, order=order)
    fmin = toff.get_knots()[0]
    fmax = toff.get_knots()[-1]
    if freqi<(1.0-eps)*fmin:
        raise Exception, 'freqi='+str(freqi)+' is lower than lowest frequency '+str(fmin)+'.'
    if freqf>(1.0+eps)*fmax:
        raise Exception, 'freqi='+str(freqf)+' is higher than highest frequency '+str(fmax)+'.'

    xi = float(toff(freqi))
    xf = float(toff(freqf))
    return window_off(h, xi, xf, win=win)
