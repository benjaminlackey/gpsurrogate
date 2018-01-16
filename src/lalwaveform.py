"""
Utilities for calling lalsimulation waveforms and converting them to
Waveform objects with either physical or geometric units.
"""

import numpy as np

import lal
import lalsimulation

from constants import *
import waveform as wave
import taylorf2

################################################################################
#                         Names of available approximants                      #
################################################################################

def print_all_approximants():
    """List all the approximants available in lalsimulation.
    """
    napprox = lalsimulation.NumApproximants
    for i in range(napprox):
        print i, lalsimulation.GetStringFromApproximant(i)


def print_td_approximants():
    """Only list the time-domain approximants in lalsimulation.
    """
    napprox = lalsimulation.NumApproximants
    for i in range(napprox):
        if lalsimulation.SimInspiralImplementedTDApproximants(i):
            print i, lalsimulation.GetStringFromApproximant(i)


def print_fd_approximants():
    """Only list the frequency-domain approximants in lalsimulation.
    """
    napprox = lalsimulation.NumApproximants
    for i in range(napprox):
        if lalsimulation.SimInspiralImplementedFDApproximants(i):
            print i, lalsimulation.GetStringFromApproximant(i)


################################################################################
#            Wrappers for lalsimulation TD and FD waveform generators          #
################################################################################

def lalsim_td_waveform(
    long_asc_nodes=0.0, eccentricity=0.0, mean_per_ano=0.0,
    phi_ref=0.0, f_ref=None,
    phase_order=-1, amplitude_order=-1, spin_order=-1, tidal_order=-1, **p):
    """Wrapper for lalsimulation.SimInspiralChooseTDWaveform.
    Simplified version of pycbc.waveform.get_td_waveform wrapper.

    Parameters
    ----------
    f_ref : Reference frequency (?For setting phi_ref? Not sure.) Defaults to f_min.
    phi_ref : Reference phase (?at f_ref?).

    Returns
    -------
    h : Waveform
    """
    if f_ref==None:
        f_ref = p['f_min']

    # Set extra arguments in the lal Dict structure
    lal_pars = lal.CreateDict()
    if phase_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNPhaseOrder(lal_pars, int(phase_order))
    if amplitude_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lal_pars, int(amplitude_order))
    if spin_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNSpinOrder(lal_pars, int(spin_order))
    if tidal_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNTidalOrder(lal_pars, int(tidal_order))
    if p['lambda1']:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, p['lambda1'])
    if p['lambda2']:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, p['lambda2'])


    # Set Approximant (C enum structure) corresponding to approximant string
    lal_approx = lalsimulation.GetApproximantFromString(p['approximant'])

    hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
        float(MSUN_SI*p['mass1']),
        float(MSUN_SI*p['mass2']),
        float(p['spin1x']), float(p['spin1y']), float(p['spin1z']),
        float(p['spin2x']), float(p['spin2y']), float(p['spin2z']),
        float(MPC_SI*p['distance']), float(p['inclination']),
        float(phi_ref),
        float(long_asc_nodes), float(eccentricity), float(mean_per_ano),
        float(p['delta_t']), float(p['f_min']), float(f_ref),
        lal_pars, lal_approx)

    # Extract data from lalsimulation's structures
    tstart = hp.epoch.gpsSeconds+hp.epoch.gpsNanoSeconds*1.0e-9
    xs = tstart + hp.deltaT*np.arange(hp.data.length)
    return wave.Waveform.from_hp_hc(xs, hp.data.data, hc.data.data)


def lalsim_fd_waveform(
    long_asc_nodes=0.0, eccentricity=0.0, mean_per_ano=0.0,
    phi_ref=0.0, f_ref=None,
    phase_order=-1, amplitude_order=-1, spin_order=-1, tidal_order=-1,
    quad1=None, quad2=None, **p):
    """Wrapper for lalsimulation.SimInspiralChooseTDWaveform.
    Simplified version of pycbc.waveform.get_td_waveform wrapper.

    Parameters
    ----------
    f_ref : Reference frequency (?For setting phi_ref? Not sure.) Defaults to f_min.
    phi_ref : Reference phase (?at f_ref?).

    Returns
    -------
    h : Waveform
    """
    if f_ref==None:
        f_ref = p['f_min']

    # Set extra arguments in the lal Dict structure
    lal_pars = lal.CreateDict()
    if phase_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNPhaseOrder(lal_pars, int(phase_order))
    if amplitude_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNAmplitudeOrder(lal_pars, int(amplitude_order))
    if spin_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNSpinOrder(lal_pars, int(spin_order))
    if tidal_order!=-1:
        lalsimulation.SimInspiralWaveformParamsInsertPNTidalOrder(lal_pars, int(tidal_order))
    if p['lambda1']:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, p['lambda1'])
    if p['lambda2']:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, p['lambda2'])
    # Add spin-induced quadrupole terms. Default is universal relations.
    # dQuadMon1 = quad1 - 1
    # dQuadMon2 = quad2 - 1
    if quad1==None:
        quad1 = taylorf2.quad_of_lambda_fit(p['lambda1'])
    if quad2==None:
        quad2 = taylorf2.quad_of_lambda_fit(p['lambda2'])
    lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(lal_pars, quad1-1.0)
    lalsimulation.SimInspiralWaveformParamsInsertdQuadMon2(lal_pars, quad2-1.0)
    print quad1, quad2
    print 'dQuadMon1 =', lalsimulation.SimInspiralWaveformParamsLookupdQuadMon1(lal_pars)
    print 'dQuadMon2 =', lalsimulation.SimInspiralWaveformParamsLookupdQuadMon2(lal_pars)

    # Set Approximant (C enum structure) corresponding to approximant string
    lal_approx = lalsimulation.GetApproximantFromString(p['approximant'])

    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
        float(MSUN_SI*p['mass1']),
        float(MSUN_SI*p['mass2']),
        float(p['spin1x']), float(p['spin1y']), float(p['spin1z']),
        float(p['spin2x']), float(p['spin2y']), float(p['spin2z']),
        float(MPC_SI*p['distance']), float(p['inclination']),
        float(phi_ref),
        float(long_asc_nodes), float(eccentricity), float(mean_per_ano),
        float(p['delta_f']), float(p['f_min']), float(p['f_max']), float(f_ref),
        lal_pars, lal_approx)

    # Extract data from lalsimulation's structures
    # The first data point in hp.data.data corresponds to f=0 not f=f_min
    fs = hp.deltaF*np.arange(hp.data.length)
    return wave.Waveform.from_hp_hc(fs, hp.data.data, hc.data.data)


################################################################################
#          lalsimulation TD and FD waveforms in dimensionless units            #
################################################################################

def dimensionless_td_waveform(
    approximant='SpinTaylorT4', q=1.0,
    spin1x=0.0, spin1y=0.0, spin1z=0.0,
    spin2x=0.0, spin2y=0.0, spin2z=0.0,
    lambda1=0.0, lambda2=0.0,
    amplitude_order=-1, phase_order=-1, spin_order=-1, tidal_order=-1,
    mf_min=0.001, delta_tbym=10.0):
    """Generate dimensionless time-domain waveforms.
    Take dimensionless arguments and return dimensionless waveform.
    Spins are defined at the reference frequency which is set to mf_lower here.

    Parameters
    ----------
    q : The small mass ratio q=m2/m1<=1 where m1 >= m2.
    spin1x, spin1y, spin1z, lambda1 : Spins and tidal parameter of the more massive star m1.
    spin2x, spin2y, spin2z, lambda2 : Spins and tidal parameter of the less massive star m2.

    Returns
    -------
    h : Waveform
        Waveform in dimensionless units.
    """
    # dimensionless -> physical:
    # Pick fiducial mtot = 1Msun and distance = 1Mpc
    # pycbc expects units of Msun and Mpc
    mtot = 1.0
    distance = 1.0
    inclination = 0.0

    mass1 = mtot / (1.0 + q)
    mass2 = mtot * q / (1.0 + q)

    f_min = C_SI**3 * mf_min / (G_SI * MSUN_SI * mtot)
    delta_t = G_SI * MSUN_SI * mtot * delta_tbym / C_SI**3

    hphys = lalsim_td_waveform(approximant=approximant,
        mass1=mass1, mass2=mass2,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        lambda1=lambda1, lambda2=lambda2,
        distance=distance, inclination=inclination,
        amplitude_order=amplitude_order, phase_order=phase_order,
        spin_order=spin_order, tidal_order=tidal_order,
        delta_t=delta_t, f_min=f_min)

    # physical -> dimensionless:
    return wave.physical_to_dimensionless_time(hphys, mtot, distance)


def dimensionless_fd_waveform(
    approximant='TaylorF2', q=1.0,
    spin1x=0.0, spin1y=0.0, spin1z=0.0,
    spin2x=0.0, spin2y=0.0, spin2z=0.0,
    lambda1=0.0, lambda2=0.0,
    amplitude_order=-1, phase_order=-1, spin_order=-1, tidal_order=-1,
    quad1=None, quad2=None,
    mf_min=0.001, mf_max=MF_ISCO, delta_mf=1.0e-6):
    """Generate dimensionless frequency-domain waveforms.
    Take dimensionless arguments and return dimensionless waveform.
    Spins are defined at the reference frequency which is set to mf_lower here.

    Parameters
    ----------
    q : The small mass ratio q=m2/m1<=1 where m1 >= m2.
    spin1x, spin1y, spin1z, lambda1 : Spins and tidal parameter of the more massive star m1.
    spin2x, spin2y, spin2z, lambda2 : Spins and tidal parameter of the less massive star m2.

    Returns
    -------
    h : Waveform
    """
    # dimensionless -> physical:
    # Pick fiducial mtot = 1Msun and distance = 1Mpc
    # pycbc expects units of Msun and Mpc
    mtot = 1.0
    distance = 1.0
    inclination = 0.0

    # Remove sections of the waveform that have an amplitude below this threshold (e.g. beginning and end).
    ampthresh = 1.0e-12

    mass1 = mtot / (1.0 + q)
    mass2 = mtot * q / (1.0 + q)

    f_min = C_SI**3 * mf_min / (G_SI * MSUN_SI * mtot)
    f_max = C_SI**3 * mf_max / (G_SI * MSUN_SI * mtot)
    delta_f = C_SI**3 * delta_mf / (G_SI * MSUN_SI * mtot)

    hphys = lalsim_fd_waveform(approximant=approximant,
        mass1=mass1, mass2=mass2,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        lambda1=lambda1, lambda2=lambda2,
        distance=distance, inclination=inclination,
        amplitude_order=amplitude_order, phase_order=phase_order,
        spin_order=spin_order, tidal_order=tidal_order,
        quad1=quad1, quad2=quad2,
        delta_f=delta_f, f_min=f_min, f_max=f_max)

    # physical -> dimensionless:
    hdim =  wave.physical_to_dimensionless_freq(hphys, mtot, distance)

    # Remove the start of the waveform that is just zeros
    # This might remove points in the middle that are nearly zero.
    # This would only be a problem if you wanted to guarantee uniform spacing.
    i_nonzero = np.where(hdim.amp>ampthresh)
    return wave.Waveform.from_amp_phase(hdim.x[i_nonzero], hdim.amp[i_nonzero], hdim.phase[i_nonzero])
