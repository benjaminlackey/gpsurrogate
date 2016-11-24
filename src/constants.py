# Math constants
import numpy as np
euler_gamma = 0.5772156649015329

# Physical constants (exact same as LAL)
G_SI = 6.67384e-11
C_SI = 299792458.0
MPC_SI = 3.085677581491367e+22
MSUN_SI = 1.9885469549614615e+30

# Dimensionless ISCO constants
MF_ISCO = 1.0/(np.pi*6.0**(3.0/2.0))
X_ISCO = 1.0/6.0
MOMEGA_ISCO = 1.0/6.0**(3.0/2.0)

# Sampling dt such that Mf_Nyquist = Mf_ISCO
DTBYM_NYQUIST_ISCO = 1.0/(2.0*MF_ISCO)


############ Convert between dimensionless and physical frequencies ############
def mf_to_f(mf, mtot):
    return mf * C_SI**3 / (G_SI * MSUN_SI * mtot)


def f_to_mf(f, mtot):
    return f * (G_SI * MSUN_SI * mtot) / C_SI**3


def f_isco(mtot):
    """ISCO GW frequency.
    mtot = mass1 + mass2 in units of M_sun.
    f_isco in Hz.
    """
    return mf_to_f(MF_ISCO, mtot)