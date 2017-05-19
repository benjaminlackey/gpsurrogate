import numpy as np
import scipy.integrate as integrate

from constants import *
import waveform as wave
import waveformset as ws
import empiricalinterpolation as eim
import taylorf2
import gaussianprocessregression as gpr
import pycbc.types

################################################################################
#                Arithmetic for amplitude of Waveform objects                  #
#                        (h1+h2, h1-h2, alpha*h, h1.h2)                        #
################################################################################


def add_amp(h1, h2):
    """Evaluate |h1|+|h2|.
    Assumes h1 and h2 have the same x values.
    """
    return wave.Waveform.from_amp_phase(h1.x, h1.amp+h2.amp, np.zeros(len(h1.x)))


def subtract_amp(h1, h2):
    """Evaluate |h1|-|h2|.
    Assumes h1 and h2 have the same x values.
    """
    return wave.Waveform.from_amp_phase(h1.x, h1.amp-h2.amp, np.zeros(len(h1.x)))


def scalar_multiply_amp(alpha, h):
    """Multiply the amplitude of h by a float alpha
    """
    return wave.Waveform.from_amp_phase(h.x, alpha*h.amp, np.zeros(len(h.x)))


def inner_product_amp(h1, h2):
    """Use Simpson's rule to
    evaluate the inner product < |h1|, |h2| > = int_xL^xH dx |h1(x)| |h2(x)|.
    Assumes h1 and h2 have the same x values.
    """
    integrand = h1.amp*h2.amp
    return integrate.simps(integrand, x=h1.x)


################################################################################
#                  Arithmetic for phase of Waveform objects                    #
#                        (h1+h2, h1-h2, alpha*h, h1.h2)                        #
################################################################################


def add_phase(h1, h2):
    """Evaluate phi1+phi2.
    Assumes h1 and h2 have the same x values.
    """
    return wave.Waveform.from_amp_phase(h1.x, np.zeros(len(h1.x)), h1.phase+h2.phase)


def subtract_phase(h1, h2):
    """Evaluate phi1-phi2.
    Assumes h1 and h2 have the same x values.
    """
    return wave.Waveform.from_amp_phase(h1.x, np.zeros(len(h1.x)), h1.phase-h2.phase)


def scalar_multiply_phase(alpha, h):
    """Multiply the phase of h by a float alpha
    """
    return wave.Waveform.from_amp_phase(h.x, np.zeros(len(h.x)), alpha*h.phase)


def inner_product_phase(h1, h2):
    """Use Simpson's rule to
    evaluate the inner product < phi1, phi2 > = int_xL^xH dx phi1(x) phi2(x).
    Assumes h1 and h2 have the same x values.
    """
    integrand = h1.phase*h2.phase
    return integrate.simps(integrand, x=h1.x)


################################################################################
#                Empirical interpolation for Waveform objects                  #
################################################################################

def empirical_interpolation_for_time_domain_waveform(waveforms, datatype):
    """Calculate empirical nodes and corresponding empirical interpolating functions
    from a set of reduced basis waveforms.

    Parameters
    ----------
    waveforms : List like set of Waveform objects
        Could be HDF5WaveformSet
    datatype : string {'amp', 'phase'}

    Returns
    -------
    empirical_node_indices : List of ints
        The indices of the empirical nodes in the Waveform objects.
    B_j : List of Waveform objects
        The empirical interpolating functions
        that are 1 at the node T_j and
        0 at the other nodes T_i (for i!=j).
    """
    nwave = len(waveforms)

    # Convert the list of Waveform objects to a list of complex numpy arrays
    if datatype == 'amp':
        wave_np = [waveforms[i].amp for i in range(nwave)]
    elif datatype == 'phase':
        wave_np = [waveforms[i].phase for i in range(nwave)]
    else:
        raise Exception, "datatype must be one of {'amp', 'phase'}."

    # Determine the empirical nodes
    empirical_node_indices = eim.generate_empirical_nodes(wave_np)

    # Determine the empirical interpolating functions B_j(t)
    B_j_np = eim.generate_interpolant_list(wave_np, empirical_node_indices)

    # Convert the arrays to Waveform objects.
    xarr = waveforms[0].x
    B_j = []
    for j in range(nwave):
        if datatype == 'amp':
            amp = B_j_np[j]
            B_j.append(wave.Waveform.from_amp_phase(xarr, amp, np.zeros(len(xarr))))
        elif datatype == 'phase':
            phase = B_j_np[j]
            B_j.append(wave.Waveform.from_amp_phase(xarr, np.zeros(len(xarr)), phase))
        else:
            raise Exception, "datatype must be one of {'amp', 'phase'}."

    return empirical_node_indices, B_j



################################################################################
#                      Reconstruct the surrogate model                         #
################################################################################

def reconstruct_amp_phase_difference(params, Bamp_j, Bphase_j, damp_gp_list, dphase_gp_list):
    """Calculate the surrogate of dh.

    Parameters
    ----------
    params : 1d array
        Waveform parameters.
    Bamp_j : List of Waveform
        List of the amplitude interpolants.
    Bphase_j : List of Waveform
        List of the phase interpolants.
    amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
    phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.

    Returns
    -------
    dh : Waveform
        Surrogate of dh.
    """
    namp_nodes = len(damp_gp_list)
    nphase_nodes = len(dphase_gp_list)

    # Calculate waveform at nodes
    amp_at_nodes = np.array([damp_gp_list[j].predict(np.atleast_2d(params))[0] for j in range(namp_nodes)])
    phase_at_nodes = np.array([dphase_gp_list[j].predict(np.atleast_2d(params))[0] for j in range(nphase_nodes)])

    # Get complex version of B_j's in array form
    Bamp_j_array = np.array([Bamp_j[j].amp for j in range(namp_nodes)])
    Bphase_j_array = np.array([Bphase_j[j].phase for j in range(nphase_nodes)])

    # Evaluate waveform
    amp_interp = np.dot(amp_at_nodes, Bamp_j_array)
    phase_interp = np.dot(phase_at_nodes, Bphase_j_array)

    # Rewrite as TimeDomainWaveform
    xarr = Bamp_j[0].x
    return wave.Waveform.from_amp_phase(xarr, amp_interp, phase_interp)


################################################################################
#                              GPSurrogate class                               #
################################################################################

def physical_to_pycbc_frequency_series(freq, h_plus, h_cross):
    """Convert numpy arrays to pycbc frequency series.
    """
    delta_f = freq[1]-freq[0]
    hp_fs = pycbc.types.FrequencySeries(h_plus, delta_f=delta_f)
    hc_fs = pycbc.types.FrequencySeries(h_cross, delta_f=delta_f)
    return hp_fs, hc_fs


class GPSurrogate(object):
    def __init__(self, Bamp, Bphase, damp_gp_list, dphase_gp_list):
        # Lists of interpolating functions
        self.Bamp = Bamp
        self.Bphase = Bphase
        # Lists of GPR functions
        self.damp_gp_list = damp_gp_list
        self.dphase_gp_list = dphase_gp_list
        # Waveform samples
        self.mf = self.Bamp[0].x
        self.mf_a = self.Bamp[0].x[0]
        self.mf_b = self.Bamp[0].x[-1]

    @classmethod
    def load(cls, Bamp_filename, Bphase_filename, damp_gp_filename, dphase_gp_filename):
        """Load surrogate model from 4 hdf5 data files.
        """
        Bamp = ws.HDF5WaveformSet(Bamp_filename)
        Bphase = ws.HDF5WaveformSet(Bphase_filename)
        damp_gp_list = gpr.load_gaussian_process_regression_list(damp_gp_filename)
        dphase_gp_list = gpr.load_gaussian_process_regression_list(dphase_gp_filename)
        return GPSurrogate(Bamp, Bphase, damp_gp_list, dphase_gp_list)

    ############# Evaluate waveform quantities in geometric units #############

    def geometric_reference_waveform(self, params, npoints=10000):
        """Reference TaylorF2 waveform in geometric units
        evaluated at the same times as the surrogate of the difference.
        """
        q, s1, s2, lambda1, lambda2 = params
        h_ref = taylorf2.dimensionless_taylorf2_waveform(
            mf=self.mf, q=q,
            spin1z=s1, spin2z=s2,
            lambda1=lambda1, lambda2=lambda2)

        # Reference waveform has zero starting phase
        h_ref.add_phase(remove_start_phase=True)
        return h_ref

    def amp_phase_difference(self, params):
        """Evaluate the surrogates for the differences \Delta\ln A and \Delta\Phi.

        Parameters
        ----------
        params : 1d array
            Waveform parameters.

        Returns
        -------
        dh : Waveform
            Waveform object with dh.amp = ln(A/A_F2) and dh.phase = DeltaPhi.
        """
        return reconstruct_amp_phase_difference(
            params, self.Bamp, self.Bphase,
            self.damp_gp_list, self.dphase_gp_list)

    def geometric_waveform(self, params):
        """Combine the reference waveform and surrogates for the differences.
        """
        # Surrogate of \Delta\Phi and \Delta\ln A
        h_diff_sur = self.amp_phase_difference(params)
        # Reference waveform
        h_ref = self.geometric_reference_waveform(params)
        # Surrogate of A and \Phi
        h_sur = h_ref.copy()
        h_sur.amp *= np.exp(h_diff_sur.amp)
        h_sur.phase += h_diff_sur.phase
        return h_sur

    ################### Evaluate waveform in physical units ##################

    def physical_waveform_zero_inclination(
        self, mass1=None, mass2=None,
        spin1z=None, spin2z=None,
        lambda1=None, lambda2=None,
        distance=None):
        """Waveform in physical units with zero inclination.
        Useful when you want a Waveform object without re-decomposing
        into amplitude and phase. This is cheap to evaluate.

        Check parameters and convert them to be used by the surrogate waveform model.
        """

        ################# Check parameter range #################
        if spin1z < -0.7 or spin1z > 0.7 or spin2z < -0.7 or spin2z > 0.7:
            raise ValueError('Valid spins: spin1z in [-0.7, 0.7], spin2z in [-0.7, 0.7]')
        if lambda1 < 0 or lambda1 > 10000 or lambda2 < 0 or lambda2 > 10000:
            raise ValueError('Valid tidal parameter range: lambda1 in [0, 10000], lambda2 in [0, 10000]')

        # If mass1 is not the larger mass, swap (mass1, mass2), (spin1z, spin2z), and (lambda1, lambda2)
        if mass1 < mass2:
            mass1, mass2 = mass2, mass1
            spin1z, spin2z = spin2z, spin1z
            lambda1, lambda2 = lambda2, lambda1

        if mass2 < 1.0:
            raise ValueError('Mass of less massive star must be >= 1M_sun.')

        q = mass2/mass1
        if q < 1.0/3.0 or q > 1.0:
            raise ValueError('Valid mass ratio range: q in [1/3, 1].')

        ########## Evaluate waveform #########
        mtot = mass1 + mass2
        params = np.array([q, spin1z, spin2z, lambda1, lambda2])
        h_geom = self.geometric_waveform(params)
        h_phys = wave.dimensionless_to_physical_freq(h_geom, mtot, distance)
        return h_phys

    def physical_waveform_lal(
        self, mass1=None, mass2=None,
        spin1z=None, spin2z=None,
        lambda1=None, lambda2=None,
        distance=None, inclination=None,
        f_min=None, f_max=None, delta_f=None,
        f_ref=None, phi_ref=None):
        """Waveform in lalsimulation format with data in arrays.
        !!!! TODO: phi_ref is the phase at f_ref. These are not currently set. !!!!

        Parameters
        ----------
        **kwargs : All arguments of self.physical_waveform_zero_inclination
        """
        ################# Check parameter range #################
        mtot = mass1 + mass2
        mf_min = f_to_mf(f_min, mtot)
        if mf_min < self.mf_a:
            raise ValueError('f_min='+str(f_min)+' < minimum frequency in surrogate ('+str(mf_to_f(self.mf_a, mtot))+').')

        ########## Evaluate waveform using LALSimulation convention #########
        #    --Uniformly spaced frequencies in [0, f_max).
        #    --Data is zero below max(f_min, first data point in h_geom) and
        #     zero above min(f_max, last data point in h_geom).

        h_phys = self.physical_waveform_zero_inclination(
            mass1=mass1, mass2=mass2,
            spin1z=spin1z, spin2z=spin2z,
            lambda1=lambda1, lambda2=lambda2,
            distance=distance)

        # Initialize arrays. The output is zero below f_min.
        freq = np.arange(0.0, f_max, delta_f)
        h_plus = np.zeros(len(freq), dtype=complex)
        h_cross = np.zeros(len(freq), dtype=complex)

        # Find the nonzero elements
        f_min_nonzero = max(f_min, h_phys.x[0])
        f_max_nonzero = min(f_max, h_phys.x[-1])
        # Can't compare arrays with 'and'. Have to use bitwise '&' instead.
        i_nonzero = np.where((freq>=f_min_nonzero) & (freq<=f_max_nonzero))
        freq_nonzero = freq[i_nonzero]

        # Amplitude and phase in the nonzero region
        amp = h_phys.interpolate('amp')(freq_nonzero)
        phase = h_phys.interpolate('phase')(freq_nonzero)

        inc_plus = 0.5*(1.0+np.cos(inclination)**2)
        inc_cross = np.cos(inclination)

        h_plus[i_nonzero] = inc_plus * 0.5*amp*np.exp(1.0j*phase)
        h_cross[i_nonzero] = inc_cross * 0.5*amp*np.exp(1.0j*(phase+np.pi/2.0))
        return freq, h_plus, h_cross

    def physical_waveform_pycbc(self, **kwargs):
        """Waveform in pycbc format

        Parameters
        ----------
        **kwargs : All arguments of self.physical_waveform_lal
        """
        freq, h_plus, h_cross = self.physical_waveform_lal(**kwargs)
        hp_fs, hc_fs = physical_to_pycbc_frequency_series(freq, h_plus, h_cross)
        return hp_fs, hc_fs
