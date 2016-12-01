import numpy as np
import scipy.integrate as integrate

import waveform as wave
import empiricalinterpolation as eim

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
#       !!!  Put the code for doing the GPR interpolation here  !!!            #
################################################################################


################################################################################
#                      Reconstruct the surrogate model                         #
################################################################################

def reconstruct_amp_phase_difference(params, Bamp_j, Bphase_j, damp_gp_list, dphase_gp_list):
    """Calculate the reduced order model waveform. This is the online stage.
    
    Parameters
    ----------
    params : 1d array
        Physical waveform parameters.
    Bamp_j : List of Waveform
        List of the ampltude interpolants.
    Bphase_j : List of Waveform
        List of the phase interpolants.
    amp_function_list : List of interpolating functions
        List of interpolating functions for the amplitude at the empirical_nodes.
    phase_function_list : List of arrays
        List of interpolating functions for the phase at the empirical_nodes.
    
    Returns
    -------
    hinterp : TimeDomainWaveform
        Reduced order model waveform
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

