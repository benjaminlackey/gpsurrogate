import numpy as np
from constants import euler_gamma
import waveform as wave

def quad_of_lambda_fit(lam):
    """Universal relation for dimensionless quadrupole in terms of dimensionless tidal parameter.
    For point particle (Kerr), quad = 1.
    For lambda >= 1, this is Eq. 15 of arXiv: 1608.02582.
    For lambda < 1, this is Sylvain Marsat's continuation.
    Copy of XLALSimUniversalRelationQuadMonVSlambda2Tidal() from LALSimUniversalRelations.c
    """
    if lam<1.0:
        b = 0.427688866723244
        c = -0.324336526985068
        d = 0.1107439432180572
        return 1.0 + b*lam + c*lam**2 + d*lam**3
    else:
        xi = np.log(lam)
        a = 0.1940
        b = 0.09163
        c = 0.04812
        d = -4.283e-3
        e = 1.245e-4
        ps = a + b*xi + c*xi**2 + d*xi**3 + e*xi**4
        return np.exp(ps)


def lamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """$\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lam1+lam2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lam1-lam2))


def deltalamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    """This is the definition found in Les Wade's paper.
    Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper.
    $\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (1.0/2.0)*(
                      np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(lam1+lam2)
                      + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(lam1-lam2)
                      )


def taylorf2_amp(mf, eta):
    """Leading order point-particle amplitude. where h(f) = h_+(f) + i h_x(f).

    Parameters
    ----------
    mf : numpy array or float
        Geometric frequency. Most efficient when mf is a numpy array.

    Returns
    -------
    amp : numpy array or float
        Amplitude of h(f) = h_+(f) + i h_x(f) in geometric units.
    """
    x = (np.pi*mf)**(2.0/3.0)
    a00 = np.sqrt( (5.0*np.pi/24.0)*eta )
    return 2.0*a00 * x**(-7./4.)


def taylorf2_amp_1pn(mf, eta):
    """1PN point-particle amplitude. where h(f) = h_+(f) + i h_x(f).
    Expression from Eq. (6.10) of arXiv:0810.5336.
    !!! This is technically wrong since you have a x**2 term (need to re-expand). !!!

    Parameters
    ----------
    mf : numpy array or float
        Geometric frequency. Most efficient when mf is a numpy array.

    Returns
    -------
    amp : numpy array or float
        Amplitude of h(f) = h_+(f) + i h_x(f) in geometric units.
    """
    x = (np.pi*mf)**(2.0/3.0)
    a00 = np.sqrt( (5.0*np.pi/24.0)*eta )
    a10 = -323./224. + 451.*eta/168.
    return 2.0*a00 * x**(-7./4.) * (1. + a10*x)


def taylorf2_phase_15pn_spin(eta, chi1, chi2):
    """2.0PN phase copied from LALSimInspiralPNCoefficients.c (line 678).
    Compute 2.0PN SS, QM, and self-spin.
    See Eq. (6.24) in arXiv:0810.5336.
    9b,c,d in arXiv:astro-ph/0504538.
    """
    # delta = (m1-m2)/Mtot
    d = np.sqrt(1.0-4.0*eta)
    # Fractional mass X_A=m_A/Mtot
    X1 = 0.5*(1.0+d)
    X2 = 0.5*(1.0-d)

    # These assumptions only hold for aligned spin:
    chi1L = chi1
    chi2L = chi2

    SL = X1**2 * chi1L + X2**2 * chi2L
    dSigmaL = d*(X2*chi2L - X1*chi1L)

    return 188.0*SL/3.0 + 25.0*dSigmaL


def taylorf2_phase_20pn_spin(eta, chi1, chi2, quad1, quad2):
    """2.0PN phase copied from LALSimInspiralPNCoefficients.c (line 678).
    Compute 2.0PN SS, QM, and self-spin.
    See Eq. (6.24) in arXiv:0810.5336.
    9b,c,d in arXiv:astro-ph/0504538.
    """
    # delta = (m1-m2)/Mtot
    d = np.sqrt(1.0-4.0*eta)
    # Fractional mass X_A=m_A/Mtot
    X1 = 0.5*(1.0+d)
    X2 = 0.5*(1.0-d)

    chi1sq = chi1**2
    chi2sq = chi2**2
    # These assumptions only hold for aligned spin:
    chi1L = chi1
    chi2L = chi2
    chi1dotchi2 = chi1*chi2

    sigma = eta * (721.0/48.0*chi1L*chi2L - 247.0/48.0*chi1dotchi2)
    sigma += (720.0*quad1 - 1.0)/96.0 * X1**2 * chi1L * chi1L
    sigma += (720.0*quad2 - 1.0)/96.0 * X2**2 * chi2L * chi2L
    sigma -= (240.0*quad1 - 7.0)/96.0 * X1**2 * chi1sq
    sigma -= (240.0*quad2 - 7.0)/96.0 * X2**2 * chi2sq

    a4 = -10.0*sigma
    return a4


def taylorf2_phase_25pn_spin(eta, chi1, chi2):
    """2.5PN phase copied from LALSimInspiralPNCoefficients.c (line 694).
    Spin-orbit terms derived from arXiv:1303.7412 (Eq. 3.15-16).
    """
    # delta = (m1-m2)/Mtot
    d = np.sqrt(1.0-4.0*eta)
    # Fractional mass X_A=m_A/Mtot
    X1 = 0.5*(1.0+d)
    X2 = 0.5*(1.0-d)

    # These assumptions only hold for aligned spin:
    chi1L = chi1
    chi2L = chi2

    SL = X1**2 * chi1L + X2**2 * chi2L
    dSigmaL = d*(X2*chi2L - X1*chi1L)

    # Spin-orbit terms can be derived from arXiv:1303.7412, Eq. 3.15-16
    gamma = (554345.0/1134.0 + 110.0*eta/9.0)*SL + (13915.0/84.0 - 10.0*eta/3.0)*dSigmaL

    a5 = -gamma
    # you need -3/2 instead of -3 because you use x instead of v**2
    a5ln = -3.0/2.0*gamma
    return a5, a5ln


def taylorf2_phase_30pn_spin(eta, chi1, chi2, quad1, quad2, spin_spin=True):
    """3.0PN phase copied from LALSimInspiralPNCoefficients.c (line 694).
    Spin-orbit terms derived from arXiv:1303.7412 (Eq. 3.15-16).
    """
    # delta = (m1-m2)/Mtot
    d = np.sqrt(1.0-4.0*eta)
    # Fractional mass X_A=m_A/Mtot
    X1 = 0.5*(1.0+d)
    X2 = 0.5*(1.0-d)

    chi1sq = chi1**2
    chi2sq = chi2**2
    # These assumptions only hold for aligned spin:
    chi1L = chi1
    chi2L = chi2

    # 3PN spin-orbit terms
    SL = X1**2 * chi1L + X2**2 * chi2L
    dSigmaL = d*(X2*chi2L - X1*chi1L)

    # 3PN spin-spin terms. These are not currently included in EOB model.
    if spin_spin==True:
        ss3 = (326.75/1.12 + 557.5/1.8*eta)*eta*chi1L*chi2L
        ss3 += ((4703.5/8.4 + 2935.0/6.0*X1 - 120.0*X1**2)*quad1 + (-4108.25/6.72 - 108.5/1.2*X1 + 125.5/3.6*X1**2)) * X1**2 * chi1sq
        ss3 += ((4703.5/8.4 + 2935.0/6.0*X2 - 120.0*X2**2)*quad2 + (-4108.25/6.72 - 108.5/1.2*X2 + 125.5/3.6*X2**2)) * X2**2 * chi2sq
    # elif spin_spin=='point':
    #     # Use the value for point particles
    #     quad1 = 1.0
    #     quad2 = 1.0
    #     ss3 = (326.75/1.12 + 557.5/1.8*eta)*eta*chi1L*chi2L
    #     ss3 += ((4703.5/8.4 + 2935.0/6.0*X1 - 120.0*X1**2)*quad1 + (-4108.25/6.72 - 108.5/1.2*X1 + 125.5/3.6*X1**2)) * X1**2 * chi1sq
    #     ss3 += ((4703.5/8.4 + 2935.0/6.0*X2 - 120.0*X2**2)*quad2 + (-4108.25/6.72 - 108.5/1.2*X2 + 125.5/3.6*X2**2)) * X2**2 * chi2sq
    else:
        ss3 = 0.0

    #print 'ss3={}'.format(ss3)

    a6 = np.pi * (3760.0*SL + 1490.0*dSigmaL)/3.0 + ss3
    return a6


def taylorf2_phase_35pn_spin(eta, chi1, chi2):
    """3.5PN phase copied from LALSimInspiralPNCoefficients.c (line 694).
    Spin-orbit terms derived from arXiv:1303.7412 (Eq. 3.15-16).
    """
    # delta = (m1-m2)/Mtot
    d = np.sqrt(1.0-4.0*eta)
    # Fractional mass X_A=m_A/Mtot
    X1 = 0.5*(1.0+d)
    X2 = 0.5*(1.0-d)

    # These assumptions only hold for aligned spin:
    chi1L = chi1
    chi2L = chi2

    SL = X1**2 * chi1L + X2**2 * chi2L
    dSigmaL = d*(X2*chi2L - X1*chi1L)

    a7 = ( -8980424995.0/762048.0 + 6586595.0*eta/756.0 - 305.0*eta**2/36.0 )*SL
    a7 -= ( 170978035.0/48384.0 - 2876425.0*eta/672.0 - 4735.0*eta**2/144.0 )*dSigmaL
    return a7


def taylorf2_phase(
    mf, tbymc, phic, eta, chi1, chi2, lambda1, lambda2,
    quad1=None, quad2=None, spin_spin=True):
    """3.5PN point-particle phase.
    FFT sign convention is $\tilde h(f) = \int h(t) e^{-2 \pi i f t} dt$
    where $h(t) = h_+(t) + i h_\times(t)$.
    """
    if quad1==None:
        quad1 = quad_of_lambda_fit(lambda1)
    if quad2==None:
        quad2 = quad_of_lambda_fit(lambda2)
    #print quad1, quad2

    tlam = lamtilde_of_eta_lam1_lam2(eta, lambda1, lambda2)
    dtlam = deltalamtilde_of_eta_lam1_lam2(eta, lambda1, lambda2)

    # Calculate the point particle coefficients
    a00 = 3.0/(128.0*eta)

    a10 = 3715.0/756.0 + 55.0*eta/9.0

    a15 = -16.0*np.pi

    a20 = 15293365.0/508032.0 + (27145.0*eta)/504.0 + (3085.0*eta**2)/72.0

    a25 = (38645.0/756.0 - 65.0*eta/9)*np.pi

    a25ln = (3.0/2.0) * (38645.0/756.0 - (65.0*eta)/9.0)*np.pi

    a30 = 11583231236531.0/4694215680.0 - (6848.0*euler_gamma)/21.0 - (640.0*np.pi**2)/3.0 - 6848.0/63.0*np.log(64.0) \
    + (-15737765635.0/3048192.0 + (2255.0*np.pi**2)/12.0)*eta \
    + (76055.0*eta**2)/1728.0 \
    - (127825.0*eta**3)/1296.0

    a30ln = -(3.0/2.0) * 6848.0/63.0

    a35 = (77096675.0/254016.0 + (378515.0*eta)/1512.0 - (74045.0*eta**2)/756.0)*np.pi

    # Add spin coefficients
    a15 += taylorf2_phase_15pn_spin(eta, chi1, chi2)
    a20 += taylorf2_phase_20pn_spin(eta, chi1, chi2, quad1, quad2)
    s25, s25ln = taylorf2_phase_25pn_spin(eta, chi1, chi2)
    a25 += s25
    a25ln += s25ln
    a30 += taylorf2_phase_30pn_spin(eta, chi1, chi2, quad1, quad2, spin_spin=spin_spin)
    a35 += taylorf2_phase_35pn_spin(eta, chi1, chi2)

    # Add the tidal coefficients
    a50 = -39.0*tlam/2.0
    a60 = -3115.0*tlam/64.0 + 6595.0*np.sqrt(1.0-4.0*eta)*dtlam/364.0

    #print a00, a10, a15, a20, a25, a30, a35
    #print a25ln, a30ln
    #print a50, a60

    # Now calculate phase for each freq
    x = (np.pi*mf)**(2.0/3.0)
    phi = -2.0*np.pi*mf*tbymc + phic + np.pi/4.0
    phi -= a00*x**(-5.0/2.0)*(1.0 + a10*x + a15*x**1.5 + a20*x**2.0 \
                             + (a25+a25ln*np.log(x))*x**2.5 \
                             + (a30+a30ln*np.log(x))*x**3.0 \
                             + a35*x**3.5 \
                             + a50*x**5 + a60*x**6)
    return phi


def dimensionless_taylorf2_waveform(
    mf=None, q=None,
    spin1x=None, spin1y=None, spin1z=None,
    spin2x=None, spin2y=None, spin2z=None,
    lambda1=None, lambda2=None,
    quad1=None, quad2=None, spin_spin=True):
    """Waveform suitable for training set.
    """
    tbymc = 0.
    phic = 0.
    eta = q/(1.+q)**2
    #amp = taylorf2_amp(mf, eta)
    # Use the 1pn amplitude
    amp = taylorf2_amp_1pn(mf, eta)
    phase = taylorf2_phase(
        mf, tbymc, phic, eta, spin1z, spin2z, lambda1, lambda2,
        quad1=quad1, quad2=quad2, spin_spin=spin_spin)
    return wave.Waveform.from_amp_phase(mf, amp, phase)
