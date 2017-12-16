#!/usr/bin/env python

import numpy as np
import lal
import lalsimulation as LS
import sys


def FD_waveform_test(Mtot, x, approximant, fLow=20.0, fHigh=16384.0, deltaF=0):
    q = 1.0/x[0]
    chi1 = x[1]
    chi2 = x[2]
    lambda1 = x[3]
    lambda2 = x[4]
    
    phiRef, fRef = 0.0, fLow
    distance, inclination = 1.0, 0.0
    
    m1 = q/(1.0+q)*Mtot
    m2 = 1.0/(1.0+q)*Mtot
    m1SI, m2SI, chi1, chi2, lambda1, lambda2, nk_max = m1*lal.MSUN_SI, m2*lal.MSUN_SI, chi1, chi2, lambda1, lambda2, -1

    longAscNodes, eccentricity, meanPerAno = 0,0,0

    LALpars = lal.CreateDict()
    LS.SimInspiralWaveformParamsInsertTidalLambda1(LALpars, lambda1)
    LS.SimInspiralWaveformParamsInsertTidalLambda2(LALpars, lambda2)

    # Nyquist frequency is set by fHigh
    # Can set deltaF = 0 to figure out required frequency spacing; the chosen deltaF depends on fLow
    
    # Documentation from LALSimInspiral.c
    #
    #  * This routine can generate TD approximants and transform them into the frequency domain.
    #  * Waveforms are generated beginning at a slightly lower starting frequency and tapers
    #  * are put in this early region so that the waveform smoothly turns on.
    #  *
    #  * If an FD approximant is used, this routine applies tapers in the frequency domain
    #  * between the slightly-lower frequency and the requested f_min.  Also, the phase of the
    #  * waveform is adjusted to introduce a time shift.  This time shift should allow the
    #  * resulting waveform to be Fourier transformed into the time domain without wrapping
    #  * the end of the waveform to the beginning.
    #  *
    #  * This routine assumes that f_max is the Nyquist frequency of a corresponding time-domain
    #  * waveform, so that deltaT = 0.5 / f_max.  If deltaF is set to 0 then this routine computes
    #  * a deltaF that is small enough to represent the Fourier transform of a time-domain waveform.
    #  * If deltaF is specified but f_max / deltaF is not a power of 2, and the waveform approximant
    #  * is a time-domain approximant, then f_max is increased so that f_max / deltaF is the next
    #  * power of 2.  (If the user wishes to discard the extra high frequency content, this must
    #  * be done separately.)
    
    hp, hc = LS.SimInspiralFD(m1SI, m2SI,
                     0.0, 0.0, chi1,
                     0.0, 0.0, chi2,
                     distance, inclination, phiRef, 
                     longAscNodes, eccentricity, meanPerAno, 
                     deltaF,
                     fLow, fHigh, fRef,
                     LALpars,
                     approximant)

    fHz = np.arange(hp.data.length)*hp.deltaF
    h = hp.data.data + 1j*hc.data.data
    
    return hp.deltaF


def _template_func(setup, func):
    """Create a timer function. Used if the "statement" is a callable."""
    def inner(_it, _timer, _func=func):
        setup()
        _t0 = _timer()
        for _i in _it:
            retval = _func()
        _t1 = _timer()
        return _t1 - _t0, retval
    return inner

def timeit_ipython(stmt, number=0, verbose=False):
    """
    Replicate ipython's %timeit functionality in a function
    Code lifted from https://github.com/ipython/ipython/blob/ea199a6ddc6cd80f46d0c106c7c7db95ebadc985/IPython/core/magic.py#L1811
    
    Added feature: grab return value of code
    
    stmt: string containing python code to time
    number <N>: execute the given statement <N> times in a loop (Default: 0 to determine automatically)
    
    Returns: 
      number of loops
      number of evaluations in loop
      best timing in seconds
      time taken to compile the code
      return value of code for best timing
    """
    import timeit
    import math
    from time import clock

    units = [u"s", u"ms",u'us',"ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    timefunc = timeit.default_timer
    repeat = timeit.default_repeat

    timer = timeit.Timer(timer=timefunc)

    # Modify the function template to return the result of the code to be timed
    template_w_return = """
def inner(_it, _timer%(init)s):
    %(setup)s
    _t0 = _timer()
    for _i in _it:
         %(stmt)s
    _t1 = _timer()
    return _t1 - _t0, retval
"""

#     src = timeit.template % {'stmt': timeit.reindent(stmt, 8), 'setup': "pass"}
    src = template_w_return % {'stmt': timeit.reindent('retval = '+stmt, 8), 'setup': "pass", 'init':"" }

    # Track compilation time so it can be reported if too long
    # Minimum time above which compilation time will be reported
    tc_min = 0.1

    t0 = clock()
    code = compile(src, "timeit_ipython", "exec")
    tc = clock()-t0

    user_ns = globals() # grab global fuction definitions and variables
    ns = {}
    exec code in user_ns, ns
    timer.inner = ns["inner"]

    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        number = 1
        for i in range(1, 10):
            t, ret = timer.timeit(number)
            if t >= 0.2:
                break
    number *= 10

    timings, rets = np.array(timer.repeat(repeat, number)).T
    best = min(timings) / number
    retval_best = rets[np.argmin(timings)]

    if best > 0.0 and best < 1000.0:
        order = min(-int(math.floor(math.log10(best)) // 3), 3)
    elif best >= 1000.0:
        order = 0
    else:
        order = 3
        
    if verbose:
        precision = 3
        print u"%d loops, best of %d: %.*g %s per loop" % (number, repeat,
                                                          precision,
                                                          best * scaling[order],
                                                          units[order])
        if tc > tc_min:
            print "Compiler time: %.2f s" % tc

    print 'Finished ', stmt, number, repeat, best, tc, retval_best
    sys.stdout.flush()

    return number, repeat, best, tc, retval_best


f_mins = np.arange(10.0, 100.0, 5)
Mtot = 2*1.35
x = np.array([1.0/1.5, 0.35, -0.12, 2000.0, 2750.0])
xnT = np.array([1.0/1.5, 0.35, -0.12, 0.0, 0.0])
xNSBH = np.array([1.0/1.5, 0.35, -0.12, 0.0, 2500.0])

tim_sur = np.array([timeit_ipython("FD_waveform_test(Mtot, x, LS.SEOBNRv4T_surrogate, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_sur', tim_sur)

tim_TF2 = np.array([timeit_ipython("FD_waveform_test(Mtot, x, LS.TaylorF2, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_TF2', tim_TF2)

tim_SEOBNRv4_ROM = np.array([timeit_ipython("FD_waveform_test(Mtot, xnT, LS.SEOBNRv4_ROM, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_SEOBNRv4_ROM', tim_SEOBNRv4_ROM)

tim_SEOBNRv4_ROM_NRTidal = np.array([timeit_ipython("FD_waveform_test(Mtot, xNSBH, LS.SEOBNRv4_ROM_NRTidal, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_SEOBNRv4_ROM_NRTidal', tim_SEOBNRv4_ROM_NRTidal)

tim_IMRPhenomD_NRTidal = np.array([timeit_ipython("FD_waveform_test(Mtot, xNSBH, LS.IMRPhenomD_NRTidal, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_IMRPhenomD_NRTidal', tim_IMRPhenomD_NRTidal)

tim_SEOBNRv4T = np.array([timeit_ipython("FD_waveform_test(Mtot, x, LS.TEOBv4, fLow=%f, fHigh=2048.0, deltaF=0)" %(f_min)) 
 for f_min in f_mins])
np.save('tim_SEOBNRv4T', tim_SEOBNRv4T)

print 'All done.'

