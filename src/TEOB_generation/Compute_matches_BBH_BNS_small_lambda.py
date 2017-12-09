import lal
import lalsimulation as LS
import numpy as np
import scipy.interpolate as ip

#------------------------------------------------------------------------------
def SEOBNRv4_ROM_NRTidal_FD(m1, m2, s1z, s2z, lambda1, lambda2,
                    f_min, f_max, deltaF=1.0/128.0,
                    delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                    approximant='SEOBNRv4_ROM_NRTidal', verbose=True):
    # print m1, m2, s1z, s2z, lambda1, lambda2, f_min, delta_t, distance, inclination
    f_ref = 0.
    phiRef = 0.

    # Must have aligned spin
    s1x, s1y, s2x, s2y = 0., 0., 0., 0.

    # Eccentricity is not part of the model
    longAscNodes = 0.
    eccentricity = 0.
    meanPerAno = 0.

    lal_approx = LS.GetApproximantFromString(approximant)

    # Insert matter parameters
    lal_params = lal.CreateDict()
    LS.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda1)
    LS.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, lambda2)

    # Evaluate FD waveform
    Hp, Hc = LS.SimInspiralFD(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI,
        s1x, s1y, s1z, s2x, s2y, s2z,
        distance*lal.PC_SI, inclination, phiRef, longAscNodes, eccentricity, meanPerAno,
        deltaF, f_min, f_max, f_ref, lal_params, lal_approx)
    f = deltaF * np.arange(Hp.data.length)
    return f, Hp, Hc

#------------------------------------------------------------------------------
def SEOBNRv4_ROM_FD(m1, m2, s1z, s2z,
                    f_min, f_max, deltaF=1.0/128.0,
                    delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                    approximant='SEOBNRv4_ROM', verbose=True):
    # print m1, m2, s1z, s2z, lambda1, lambda2, f_min, delta_t, distance, inclination
    f_ref = 0.
    phiRef = 0.

    # Must have aligned spin
    s1x, s1y, s2x, s2y = 0., 0., 0., 0.

    # Eccentricity is not part of the model
    longAscNodes = 0.
    eccentricity = 0.
    meanPerAno = 0.

    lal_approx = LS.GetApproximantFromString(approximant)

    lal_params = lal.CreateDict()

    # Evaluate FD waveform
    Hp, Hc = LS.SimInspiralFD(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI,
        s1x, s1y, s1z, s2x, s2y, s2z,
        distance*lal.PC_SI, inclination, phiRef, longAscNodes, eccentricity, meanPerAno,
        deltaF, f_min, f_max, f_ref, lal_params, lal_approx)
    f = deltaF * np.arange(Hp.data.length)
    return f, Hp, Hc

#------------------------------------------------------------------------------
def match(h1, h2, psdfun, deltaF, zpf=2, verbose=True):
    """
    Compute the match between FD waveforms h1, h2

    :param h1, h2: data from frequency series [which start at f=0Hz]
    :param psdfun: power spectral density as a function of frequency in Hz
    :param zpf:    zero-padding factor
    """
    assert(len(h1) == len(h2))
    if len(h1) > 250000:
        print 'n > 250000. Match calculation could take a very long time!'
    n = len(h1)
    f = deltaF*np.arange(0,n)
    psd_ratio = psdfun(100) / np.array(map(psdfun, f))
    psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan
    h1abs = np.abs(h1)
    h2abs = np.abs(h2)
    norm1 = np.dot(h1abs, h1abs*psd_ratio)
    norm2 = np.dot(h2abs, h2abs*psd_ratio)
    integrand = h1 * h2.conj() * psd_ratio # different name!
    integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)])
    if verbose: print 'match: len(zpadded integrand)', len(integrand_zp)
    #integrand_zp = np.lib.pad(integrand, n*zpf, 'constant', constant_values=0) # zeropad it
    csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr; numpy.fft = Mma iFFT with our conventions
    return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2)

#------------------------------------------------------------------------------
def match_interpolated_for_range(f1, f2, h1, h2, psdfun, flo, fhi, df, zpf=5):
    """
    Compute the match between FD waveforms h1, h2 interpolated onto a prescribed grid

    :param h1, h2: arrays
    :param psdfun: power spectral density as a function of frequency in Hz
    :param zpf:    zero-padding factor
    """
#     f1 = df1*np.arange(0,len(h1))
#     f2 = df2*np.arange(0,len(h2))
    A1 = np.abs(h1)
    A2 = np.abs(h2)
    A1_I = ip.InterpolatedUnivariateSpline(f1, A1, k=3)
    A2_I = ip.InterpolatedUnivariateSpline(f2, A2, k=3)
    phi1 = np.unwrap(np.angle(h1))
    phi2 = np.unwrap(np.angle(h2))
    phi1_I = ip.InterpolatedUnivariateSpline(f1, phi1, k=3)
    phi2_I = ip.InterpolatedUnivariateSpline(f2, phi2, k=3)
    if ((fhi > f1[-1]) or (fhi > f2[-1])):
        fhi = np.min([f1[-1], f2[-1]])
        print 'fhi > maximum frequency of data! Resetting fhi to', fhi

    n = int((fhi - flo) / df)
    f = flo + np.arange(n) * df
    h1_data = A1_I(f)*np.exp(1j*phi1_I(f))
    h2_data = A2_I(f)*np.exp(1j*phi2_I(f))

    psd_ratio = psdfun(100) / np.array(map(psdfun, f))
    if f[0] == 0:
        psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan

    h1abs = np.abs(h1_data)
    h2abs = np.abs(h2_data)
    norm1 = np.dot(h1abs, h1abs*psd_ratio)
    norm2 = np.dot(h2abs, h2abs*psd_ratio)
    integrand = h1_data * h2_data.conj() * psd_ratio # different name!
    #integrand_zp = np.lib.pad(integrand, n*zpf, 'constant', constant_values=0) # zeropad it
    integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)]) # zeropad it, in case we don't have np.lib.pad
    csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr; numpy.fft = Mma iFFT with our conventions
    return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2)

#------------------------------------------------------------------------------
def compute_match(lambda_val=10.0, q=1.5):
    Mtot = 2.8
    m1, m2 = Mtot*q/(1.+q), Mtot/(1.+q)
    s1z, s2z = 0.05, 0.05
    lambda1, lambda2 = lambda_val, lambda_val
    f_min = 20.0
    f_max = 8192.0
    deltaF = 1.0/128.0

    f1, Hp1, Hc1 = SEOBNRv4_ROM_NRTidal_FD(m1, m2, s1z, s2z, lambda1, lambda2,
                        f_min, f_max, deltaF=deltaF,
                        delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                        approximant='SEOBNRv4_ROM_NRTidal')

    f2, Hp2, Hc2 = SEOBNRv4_ROM_FD(m1, m2, s1z, s2z,
                        f_min, f_max, deltaF=deltaF,
                        delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                        approximant='SEOBNRv4_ROM')

    deltaF_match = 0.01
    m = match_interpolated_for_range(f1, f2,
                         Hp1.data.data, Hp2.data.data,
                         LS.SimNoisePSDaLIGOZeroDetHighPower,
                         f_min, f_max, deltaF_match, zpf=5)

    #m = match(Hp1.data.data, Hp2.data.data, LS.SimNoisePSDaLIGOZeroDetHighPower, deltaF, zpf=2)
    return m

lambdas = np.logspace(-2.0, 3.0, 50)
matches = np.array([compute_match(lambda_val=l, q=1.5) for l in lambdas])

print lambdas, matches
np.savetxt('matches_BBH_BNS_small_lambda.dat', np.vstack([lambdas, matches]).T)

