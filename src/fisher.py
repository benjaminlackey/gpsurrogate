import numpy as np

import scipy.interpolate as interpolate
import scipy.integrate as integrate

import waveform as wave

def waveform_finite_difference(h1, h2, theta1, theta2, dtype='central'):
    """Approximate dh/dtheta with finite differencing.

    Parameters
    ----------
    h1, h2 : Waveform
        Both waveforms must have identical spacing.
    theta1, theta2 : float
        Parameter values corresponding to h1, h2.
    dtype : {'forward', 'central', 'backward'}
        Type of finite differencing. Only use 'forward' or 'backward' if
        you are at the boundary of your parameter space.
    """
    lnamp1 = np.log(h1.amp)
    lnamp2 = np.log(h2.amp)

    # amp, phase derivatives
    # Basically the same formula for (forward, central, backward).
    # Accuracy is 1st order for (forward, backward) and 2nd order for central.
    dlnamp_dtheta = (lnamp2 - lnamp1) / (theta2 - theta1)
    dphase_dtheta = (h2.phase - h1.phase) / (theta2 - theta1)

    # Choose which waveform corresponds to the point in parameter space
    # you want to calculate the derivative at
    if dtype=='central':
        # Use the interpolated waveform at the midpoint
        lnamp = 0.5*(lnamp1 + lnamp2)
        phase = 0.5*(h1.phase + h2.phase)
    elif dtype=='forward':
        # Use the waveform corresponding to theta1
        lnamp = lnamp1
        phase = h1.phase
    elif dtype=='backward':
        # Use the waveform corresponding to theta2
        lnamp = lnamp2
        phase = h2.phase
    else:
        raise Exception, "dtype options are {'forward', 'central', 'backward'}."

    # Waveform derivative
    dhbyh_comp = dlnamp_dtheta + 1.0j*dphase_dtheta
    dhbyh_amp, dhbyh_phase = wave.complex_to_amp_phase(dhbyh_comp)
    dh_amp = np.exp(lnamp)*dhbyh_amp
    dh_phase = phase + dhbyh_phase
    return wave.Waveform.from_amp_phase(h1.x, dh_amp, dh_phase)


def inner_product(h1, h2, psd):
    """Evaluate (h1|h2).
    TODO: Allow optional f_min, f_max arguments?

    Parameters
    ----------
    h1, h2 : Waveform
    psd : 1d-array of PSD
    """
    integrand_real = h1.amp*h2.amp*np.cos(h1.phase-h2.phase) / psd
    return 4.0*integrate.simps(integrand_real, x=h1.x)


class NumericalFisher(object):
    """Fisher matrix analysis of errors using numerical derivatives.
    """
    def __init__(self, waveform, params, f_min, f_max, psd_array):
        """
        Parameters
        ----------
        waveform : func(params, f_min, f_max)
            Returns Waveform object
        params : list
        f_min : float
        f_max : float
        psd : 2d-array with shape (n, 2).
            Columns are (freq, psd).
        """
        self.waveform = waveform
        # list() creates a copy of the data from a list-like object (list, tuple, np.array, etc.)
        self.params = list(params)
        self.f_min = f_min
        self.f_max = f_max

        # PSD function
        # Use linear interpolation to capture spikes
        self.psdoff = interpolate.UnivariateSpline(psd_array[:, 0], psd_array[:, 1], k=1, s=0)

        # Lists and matrices of expensive quantities that are reused
        self.derivatives = None
        self.fisher = None
        self.prior = None
        self.cov = None

    ######## Single item functions ########

    def waveform_eval(self, params):
        """Evaluate the waveform with the parameters params.
        """
        return self.waveform(params, self.f_min, self.f_max)

    def derivative_eval(self, i, dtheta, dtype='central'):
        """Evaluate the waveform derivatives with central differencing.

        TODO: have a type option to choose between (central, left, right) differencing
        for when you are near the bounds of where you can calculate waveforms (e.g. eta=0.25).
        """
        # Make copies not references with list()
        params1 = list(self.params)
        params2 = list(self.params)

        if dtype=='central':
            # Shift (params1, params2) so they are centered on params
            params1[i] -= dtheta/2
            params2[i] += dtheta/2
        elif dtype=='forward':
            # Shift params2 to be larger than params by dtheta
            params2[i] += dtheta
        elif dtype=='backward':
            # Shift params1 to be less than params by dtheta
            params1[i] -= dtheta
        else:
            raise Exception, "dtype options are {'forward', 'central', 'backward'}."

        theta1 = params1[i]
        theta2 = params2[i]
        h1 = self.waveform_eval(params1)
        h2 = self.waveform_eval(params2)
        dh = waveform_finite_difference(h1, h2, theta1, theta2, dtype=dtype)
        return dh

    def snr(self):
        """Optimal SNR.
        """
        h = self.waveform_eval(self.params)
        # Resample PSD to match waveform samples
        fs = h.x
        psd_resamp = self.psdoff(fs)
        return np.sqrt(inner_product(h, h, psd_resamp))

    ########### Evaluate arrays ###########

    def derivative_list_eval(self, param_indices, dthetas, dtypes):
        """Calculate the derivatives for the parameters with indices param_indices,
        and store them as a list.

        Parameters
        ----------
        param_indices : list of int
            Indices of the parameters you want in the Fisher matrix.
        dthetas : list of floats
            With for finite difference for *ALL* parameters.
            Use anything (e.g. 0) if not included in Fisher matrix.
            !!!!!!!!! TODO: This is dumb. You should only have to specify the ones that are used. !!!!!!!!!
        dtypes : list of strings
            Can choose from {'forward', 'central', 'backward'}.
            Type of finite differencing. Only use 'forward' or 'backward' if
            you are at the boundary of your parameter space.
        """
        self.derivatives = []
        nparams = len(param_indices)
        fisher = np.zeros((nparams, nparams))
        for i in range(nparams):
            pi = param_indices[i]
            dtheta_pi = dthetas[pi]
            dh_pi = self.derivative_eval(pi, dtheta_pi, dtype=dtypes[i])
            self.derivatives.append(dh_pi)

    def fisher_matrix(self):
        """Calculate the Fisher matrix.
        """
        # Resample PSD to match waveform samples
        fs = self.derivatives[0].x
        psd_resamp = self.psdoff(fs)

        nparams = len(self.derivatives)
        self.fisher = np.zeros((nparams, nparams))
        # Fisher matrix is symmetric so only calculate lower triangle
        for i in range(nparams):
            for j in range(i+1):
                dh_i = self.derivatives[i]
                dh_j = self.derivatives[j]
                gamma_ij = inner_product(dh_i, dh_j, psd_resamp)
                self.fisher[i, j] = gamma_ij
                self.fisher[j, i] = gamma_ij
        return self.fisher

    def prior_matrix(self, sigmas):
        """Set the prior matrix.
        (Inverse of the covariance matrix for a multivariate Gaussian prior.)

        Parameters
        ----------
        sigmas : List of (1-sigma prior width or None) for each parameter.
            Ex: [0.1, 10, None, 0.9, None, None]
        """
        prior_diag = np.zeros(len(sigmas))
        for i in range(len(sigmas)):
            if sigmas[i]==None:
                prior_diag[i] = 0.0
            else:
                prior_diag[i] = 1./sigmas[i]**2
        self.prior = np.diag(prior_diag)

    def covariance_matrix(self):
        """Calculate the covariance matrix.
        Cov = (Fisher + Prior)^(-1)
        """
        self.cov = np.linalg.inv(self.fisher + self.prior)
        return self.cov

    def systematic_error(self, h_exact):
        """Bias in each parameter when the waveform model doesn't agree with the exact waveform h_exact.
        """
        # Difference between h_exact and h:
        # 1. Resample h_exact to match h.
        # 2. Reexpress two waveforms as complex arrays and subtract them.
        # 3. Store the difference as a Waveform.
        h = self.waveform_eval(self.params)
        h_exact_resamp = h_exact.copy()
        h_exact_resamp.resample(h.x)
        h_comp = h.complex()
        h_exact_resamp_comp = h_exact_resamp.complex()
        delta_h_comp = h_comp - h_exact_resamp_comp
        delta_h = wave.Waveform.from_complex(h.x, delta_h_comp)

        # Resample PSD to match waveform samples
        fs = self.derivatives[0].x
        psd_resamp = self.psdoff(fs)

        nparams = len(self.derivatives)
        diff_dot_deriv = np.zeros(nparams)
        for j in range(nparams):
            dh_j = self.derivatives[j]
            inner = inner_product(delta_h, dh_j, psd_resamp)
            diff_dot_deriv[j] = inner

        # matrix*vector to get vector of systematic errors
        return -np.dot(self.cov, diff_dot_deriv)

    def get_sigmas(self):
        """1-sigma uncertainties in each parameter.
        """
        return self.cov.diagonal()**0.5

    def get_correlation_matrix(self):
        """Correlation matrix between parameters.
        """
        sigmas = self.get_sigmas()
        nparams = len(sigmas)
        corr = np.array([[self.cov[i, j]/(sigmas[i]*sigmas[j]) for j in range(nparams)] for i in range(nparams)])
        return corr

    def plot_derivative(i):
        pass

    def plot_fisher_integrand(i, j):
        pass


################################################################################
#            Functions for converting between standard parameters              #
#                    and Fisher matrix parameters.                             #
################################################################################


# Mchirp and eta do not depend on which mass (m1 or m2) is greater.
# Going backwards requires a choice for which mass is greater.
def mchirp_of_m1_m2(m1, m2):
    return (m1*m2)**(3.0/5.0) / (m1+m2)**(1.0/5.0)

def eta_of_m1_m2(m1, m2):
    return (m1*m2) / (m1+m2)**2.0

def eta_of_q(q):
    """Takes either big Q=m_1/m_2 or little q=m_2/m_1 and returns
    symmetric mass ratio eta.
    """
    return q / (1.0 + q)**2

def big_and_small_q_of_eta(eta):
    bigq = (1.0-2.0*eta + np.sqrt(1.0-4.0*eta))/(2.0*eta)
    smallq = (1.0-2.0*eta - np.sqrt(1.0-4.0*eta))/(2.0*eta)

    return bigq, smallq

def m1_of_mchirp_eta(mchirp, eta):
    """m1 is always the more massive star (the primary)
    """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 + np.sqrt(1.0-4.0*eta))


def m2_of_mchirp_eta(mchirp, eta):
    """m2 is always the less massive star (the secondary)
    """
    return (1.0/2.0)*mchirp*eta**(-3.0/5.0) * (1.0 - np.sqrt(1.0-4.0*eta))

def ssym_of_s1_s2(s1, s2):
    return 0.5*(s1+s2)

def santisym_of_s1_s2(s1, s2):
    return 0.5*(s1-s2)

def s1_s2_of_pe_params(ssym, santi):
    s1 = ssym + santi
    s2 = ssym - santi
    return s1, s2

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

def lam1_lam2_of_pe_params(eta, lamt, dlamt):
    """lam1 is for the the primary mass m_1.
    lam2 is for the the secondary mass m_2.
    m_1 >= m2.
    """

    a = (8.0/13.0)*(1.0+7.0*eta-31.0*eta**2)
    b = (8.0/13.0)*np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)
    c = (1.0/2.0)*np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)
    d = (1.0/2.0)*(1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)

    den = (a+b)*(c-d) - (a-b)*(c+d)

    lam1 = ( (c-d)*lamt - (a-b)*dlamt )/den
    lam2 = (-(c+d)*lamt + (a+b)*dlamt )/den

    return lam1, lam2
