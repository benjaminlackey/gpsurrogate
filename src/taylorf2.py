import numpy as np
from constants import euler_gamma
import waveform as wave

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


def taylort4_xdot(mfreq, eta):
    """3.5PN point-particle amplitude.
    FFT sign convention is $\tilde h(f) = \int h(t) e^{-2 \pi i f t} dt$
    where $h(t) = h_+(t) + i h_\times(t)$. 
    """    
    # Calculate the coefficients once
    xdot00 = (64.0*eta)/5.0
    
    xdot10 = -( (743.0 + 924.0*eta)/336.0 )

    xdot15 = 4.0*np.pi

    xdot20 = 34103.0/18144.0 + (13661.0*eta)/2016.0 + (59.0*eta**2)/18.0

    xdot25 = -( 4159.0/672.0 + 15876.0*eta/672.0 )*np.pi

    xdot30 = 16447322263.0/139708800.0 - (1712.0*euler_gamma)/105.0 \
    + (16.0*np.pi**2)/3.0 - (856.0*np.log(16.0))/105.0 \
    - ( 56198689.0/217728.0 - (451.0*np.pi**2/48.0) )*eta \
    + (541.0*eta**2)/896.0 \
    - (5605*eta**3)/2592.0

    xdot30ln = -(856.0/105.0)

    xdot35 = -( 4415.0/4032.0 - (358675.0*eta)/6048.0 - (91495*eta**2)/1512.0 )*np.pi
    
    # Now calculate phase for each freq
    x = (np.pi*mfreq)**(2.0/3.0)

    xdot = xdot00*x**5 * (1.0 + xdot10*x + xdot15*x**1.5 + xdot20*x**2 + xdot25*x**2.5 \
                          + (xdot30+xdot30ln*np.log(x))*x**3 + xdot35*x**3.5)
    return xdot


def taylorf2_amp(mfreq, eta):
    """3.5PN point-particle amplitude.
    FFT sign convention is $\tilde h(f) = \int h(t) e^{-2 \pi i f t} dt$
    where $h(t) = h_+(t) + i h_\times(t)$.
    """
    # Calculate the coefficients once
    # !!!!!!! The imaginary parts of the coefficients correspond to
    # using h = h_+ - i h_x (not h_+ + i h_x). This doesn't really 
    # matter since you take the absolute value, but you should really
    # correct it to make things cleaner. !!!!!!!
    a00 = -8.0*np.sqrt(np.pi/5.0)*eta
    
    a10 = -(107.0/42.0) + (55.0*eta)/42.0
    
    a15 = 2.0*np.pi
    
    a20 = -(2173.0/1512.0) - (1069.0*eta)/216.0 + (2047.0*eta**2/1512.0)
    
    a25 = -(107.0*np.pi)/21.0 + ( (34.0*np.pi)/21.0 - 24.0j )*eta
    
    a30 = 27027409.0/646800.0 - (856.0*euler_gamma)/105.0 + (2*np.pi**2)/3.0 \
    - (428.0*np.log(16.0))/105.0 + (428.0j*np.pi)/105.0 \
    + ( -278185.0/33264.0 + (41.0*np.pi**2)/96.0 )*eta \
    - (20261.0*eta**2)/2772.0 \
    + (114635.0*eta**3)/99792.0
    
    a30ln = -428.0/105.0
    
    # Now calculate amplitude for each freq
    # Use TaylorT4 value for xdot
    # A22 has a very small phase component. Ignore it by taking the absolute value.
    x = (np.pi*mfreq)**(2.0/3.0)
    A22 = a00*x*( 1.0 + a10*x + a15*x**1.5 + a20*x**2 + a25*x**2.5 + (a30+a30ln*np.log(x))*x**3 )
    xdot = taylort4_xdot(mfreq, eta)
    A22tilde = np.abs(A22) * np.sqrt( 2.0 * np.pi / (3.0 * x**0.5 * xdot) )
    
    # Convert spherical harmonic coefficient A22 to h
    return 0.5*np.sqrt(5.0/np.pi) * A22tilde


def taylorf2_phase(mfreq, tbymc, phic, eta, chi1, chi2, lambda1, lambda2):
    """3.5PN point-particle phase.
    FFT sign convention is $\tilde h(f) = \int h(t) e^{-2 \pi i f t} dt$
    where $h(t) = h_+(t) + i h_\times(t)$.
    """
    delta = np.sqrt(1.0-4.0*eta)
    
    chis = 0.5*(chi1 + chi2)
    chia = 0.5*(chi1 - chi2)
    
    tlam = lamtilde_of_eta_lam1_lam2(eta, lambda1, lambda2)
    dtlam = deltalamtilde_of_eta_lam1_lam2(eta, lambda1, lambda2)
    
    beta = (113.0/12.0 - 19.0*eta/3.0)*chis + (113.0/12.0)*delta*chia
    
    sigma = eta*( (721.0/48.0)*(chis**2 - chia**2) - (247.0/48.0)*(chis**2 - chia**2) )\
    + (1.0 - 2.0*eta)*( (719.0/96.0)*(chis**2 + chia**2) - (233.0/96.0)*(chis**2 + chia**2) )\
    + delta*( (719.0/48.0)*(chis*chia) - (233.0/48.0)*(chis*chia) )
             
    gamma = (732985.0/2268.0 - 24260.0*eta/81.0 - 340.0*eta**2/9.0)*chis\
    + (732985.0/2268.0 + 140.0*eta/9.0)*delta*chia
    
    p6 = np.pi*( (2270.0/3.0)*delta*chia + (2270.0/3.0-520.0*eta)*chis )\
    + (75515.0/144.0 - 8225.0*eta/18.0)*delta*chis*chia\
    + (75515.0/288.0 - 263245.0*eta/252.0 - 480.0*eta**2)*chia**2\
    + (75515.0/288.0 - 232415.0*eta/504.0 + 1255.0*eta**2/9.0)*chis**2
    
    p7 = (-25150083775.0/3048192.0 + 26804935.0*eta/6040.0 - 1985.0*eta**2/48.0)*delta*chia\
    + (-25150083775.0/3048192.0 + 10566655595.0*eta/762048.0 - 1042165.0*eta**2/3024.0 + 5345.0*eta**3/36.0)*chis\
    + (14585.0/24.0 - 2380.0*eta)*delta*chia**3\
    + (14585.0/24.0 - 475.0*eta/6.0 + 100.0*eta**2/3.0)*chis**3\
    + (14585.0/8.0 - 215.0*eta/2.0)*delta*chia*chis**2\
    + (14585.0/8.0 - 7270.0*eta + 80.0*eta**2)*chia**2*chis
    
    p8 = np.pi*( (233915.0/168.0 - 99185.0*eta/252.0)*delta*chia\
                + (233915.0/168.0 - 3970375.0*eta/2268.0 + 19655.0*eta**2/189.0) )*chis
    
    # Calculate the coefficients once
    a00 = 3.0/(128.0*eta)
    
    a10 = 3715.0/756.0 + 55.0*eta/9.0
    
    a15 = -16.0*np.pi + 4.0*beta
    
    a20 = 15293365.0/508032.0 + (27145.0*eta)/504.0 + (3085.0*eta**2)/72.0 - 10.0*sigma
    
    a25 = (38645.0/756.0 - 65.0*eta/9)*np.pi - gamma
    
    a25ln = (3.0/2.0) * (38645.0/756.0 - (65.0*eta)/9.0)*np.pi - (3.0/2.0)*gamma
    
    a30 = 11583231236531.0/4694215680.0 - (6848.0*euler_gamma)/21.0 - (640.0*np.pi**2)/3.0 - 6848.0/63.0*np.log(64.0) \
    + (-15737765635.0/3048192.0 + (2255.0*np.pi**2)/12.0)*eta \
    + (76055.0*eta**2)/1728.0 \
    - (127825.0*eta**3)/1296.0 + p6
    
    a30ln = -(3.0/2.0) * 6848.0/63.0
    
    a35 = (77096675.0/254016.0 + (378515.0*eta)/1512.0 - (74045.0*eta**2)/756.0)*np.pi + p7
    
    #a40 = p8
    
    #a40ln = -(3.0/2.0)*p8
    
    a40 = 0.0
    a40ln = 0.0
    
    a50 = -39.0*tlam/2.0
    
    a60 = -3115.0*tlam/64.0 + 6595.0*np.sqrt(1.0-4.0*eta)*dtlam/364.0
    
    # Now calculate phase for each freq
    x = (np.pi*mfreq)**(2.0/3.0)
    
    phi = -2.0*np.pi*mfreq*tbymc + phic + np.pi/4.0 \
    - a00*x**(-5.0/2.0)*(1.0 + a10*x + a15*x**1.5 + a20*x**2.0 \
                         + (a25+a25ln*np.log(x))*x**2.5 \
                         + (a30+a30ln*np.log(x))*x**3.0 \
                         + a35*x**3.5 \
                         + (a40+a40ln*np.log(x))*x**4.0 \
                         + a50*x**5 + a60*x**6)
    return phi


def taylorf2(tbymc, phic, eta, chi1, chi2, lambda1, lambda2, mf_lower, mf_upper, delta_mf):
    """Calculate 3.5PN TaylorF2 waveform in dimensionless units.
    FFT sign convention is $\tilde h(f) = \int h(t) e^{-2 \pi i f t} dt$
    where $h(t) = h_+(t) + i h_\times(t)$.
    """
    mfreq = np.arange(mf_lower, mf_upper, delta_mf)
    amp = taylorf2_amp(mfreq, eta)
    phase = taylorf2_phase(mfreq, tbymc, phic, eta, chi1, chi2, lambda1, lambda2)
    return wave.Waveform.from_amp_phase(mfreq, amp, phase)

