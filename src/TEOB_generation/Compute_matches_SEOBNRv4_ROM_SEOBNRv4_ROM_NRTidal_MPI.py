#!/usr/bin/env python

# coding: utf-8

import lal
import lalsimulation as LS

import numpy as np
import shutil, os, sys
import traceback
from subprocess import check_call
import scipy
import scipy.interpolate as ip
from scipy import optimize
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import os.path
import h5py
import glob

from mpi4py import MPI
import warnings
import json
from multiprocessing import Process, Queue

import romspline

from tqdm import *

# MP 08 / 2017

#------------------------------------------------------------------------------
def spin_tidal_eob_FD(m1, m2, s1z, s2z, lambda1, lambda2,
                    f_min, f_max, deltaF=1.0/128.0,
                    delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                    approximant='TEOBv4', verbose=True):
    """EOB waveform with aligned spin and tidal interactions. 
    l=3 tidal interaction and l=2,3 f-mode calculated with universal relations.
    
    Parameters
    ----------
    approximant : 'TEOBv2' or 'TEOBv4', 'SEOBNRv4_ROM_NRTidal'
    
    Returns
    -------
    Waveform object
    """
    # print m1, m2, s1z, s2z, lambda1, lambda2, f_min, delta_t, distance, inclination
    f_ref = 0.
    phiRef = 0.

    # Must have aligned spin
    s1x, s1y, s2x, s2y = 0., 0., 0., 0.

    # Eccentricity is not part of the model
    longAscNodes = 0.
    eccentricity = 0.
    meanPerAno = 0.

    # Set the EOB approximant
    if (approximant not in ['TEOBv2', 'TEOBv4', 'SEOBNRv4_ROM_NRTidal']):
        raise Exception, "Approximant must be 'TEOBv2', 'TEOBv4' or 'SEOBNRv4_ROM_NRTidal'."
    lal_approx = LS.GetApproximantFromString(approximant)

    if (approximant in ['TEOBv2', 'TEOBv4']):
        # Calculate higher order matter effects from universal relations
        # lambda3 given in terms of lambda2
        lambda31_ur = LS.SimUniversalRelationlambda3TidalVSlambda2Tidal(lambda1)
        lambda32_ur = LS.SimUniversalRelationlambda3TidalVSlambda2Tidal(lambda2)
        # Omega2 given in terms of lambda2
        omega21_ur = LS.SimUniversalRelationomega02TidalVSlambda2Tidal(lambda1)
        omega22_ur = LS.SimUniversalRelationomega02TidalVSlambda2Tidal(lambda2)
        # Omega3 given in terms of lambda3 (not lambda2)
        omega31_ur = LS.SimUniversalRelationomega03TidalVSlambda3Tidal(lambda31_ur)
        omega32_ur = LS.SimUniversalRelationomega03TidalVSlambda3Tidal(lambda32_ur)

    # Insert matter parameters
    lal_params = lal.CreateDict()
    LS.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda1)
    LS.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, lambda2)
    if (approximant in ['TEOBv2', 'TEOBv4']):
        LS.SimInspiralWaveformParamsInsertTidalOctupolarLambda1(lal_params, lambda31_ur)
        LS.SimInspiralWaveformParamsInsertTidalOctupolarLambda2(lal_params, lambda32_ur)
        LS.SimInspiralWaveformParamsInsertTidalQuadrupolarFMode1(lal_params, omega21_ur)
        LS.SimInspiralWaveformParamsInsertTidalQuadrupolarFMode2(lal_params, omega22_ur)
        LS.SimInspiralWaveformParamsInsertTidalOctupolarFMode1(lal_params, omega31_ur)
        LS.SimInspiralWaveformParamsInsertTidalOctupolarFMode2(lal_params, omega32_ur)
    
    # Evaluate FD waveform
    Hp, Hc = LS.SimInspiralFD(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI,
        s1x, s1y, s1z, s2x, s2y, s2z,
        distance*lal.PC_SI, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, 
        deltaF, f_min, f_max, f_ref, lal_params, lal_approx)
    f = deltaF * np.arange(Hp.data.length)
    return f, Hp, Hc

#------------------------------------------------------------------------------
def spin_tidal_IMRPhenomD_FD(m1, m2, s1z, s2z, lambda1, lambda2,
                    f_min, f_max, deltaF=1.0/128.0,
                    delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                    approximant='IMRPhenomD_NRTidal', verbose=True):
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
def BNS_tidal_match(i, m1, m2, chi1, chi2, lambda1, lambda2, tmpdir,
                    f_min=40.0, f_max=8192.0, deltaF=1.0/128.0, deltaF_match=0.1, allow_skip=True):
    args = i, m1, m2, chi1, chi2, lambda1, lambda2, f_min, f_max, deltaF
    print '*** Starting BNS_tidal_match for parameters:', args
  
    if os.path.isfile(tmpdir+'match_'+str(i)+'.npy') and allow_skip:
        print '*** Skipping existing configuration for parameters:', args
        return
 
    distance=1.0e6
    inclination=0.0

    try:
#        fP, HpP, HcP = spin_tidal_IMRPhenomD_FD(m1, m2, chi1, chi2, lambda1, lambda2,
#                        f_min, f_max, deltaF=deltaF,
#                        distance=distance, inclination=inclination,
#                        approximant='IMRPhenomD_NRTidal')
        fP, HpP, HcP = SEOBNRv4_ROM_FD(m1, m2, chi1, chi2,
                        f_min, f_max, deltaF=deltaF,
                        distance=distance, inclination=inclination,
                        approximant='SEOBNRv4_ROM')

        fE, HpE, HcE = spin_tidal_eob_FD(m1, m2, chi1, chi2, lambda1, lambda2,
                        f_min, f_max, deltaF=deltaF,
                        distance=distance, inclination=inclination,
                        approximant='SEOBNRv4_ROM_NRTidal')
        # The frequency resolution may be different between the two approximants, we interpolate in the match function

        m = match_interpolated_for_range(fP, fE, 
                             HpP.data.data, HpE.data.data, 
                             LS.SimNoisePSDaLIGOZeroDetHighPower, 
                             f_min, f_max, deltaF_match, zpf=5)

        np.save(tmpdir+'match_'+str(i)+'.npy', np.array([m1, m2, chi1, chi2, lambda1, lambda2, m]))
        #np.save(tmpdir+'wfs_'+str(i)+'.npy', np.array([fE, HpP.data.data, HpE.data.data]))

        print '*** BNS_tidal_match finished for parameters:', args
    except Exception as e:
        print '***********************************************************************'
        print '*** BNS_tidal_match failed for parameters:', args
        print '*** Error %s' % e
        print '***********************************************************************'

        f = open('FAILED_BNS_tidal_match_PARAMS.txt', 'a')
        print args
        s = '%.16e'%(args[1])
        for arg in args[2:]:
            if (type(arg) == np.unicode) or (type(arg) == str):
                s += ' '+str(arg)
            elif type(arg) == np.ndarray:
                for a in arg:
                    s += ' %.16e'%(a)
            else:
                s += ' %.16e'%(arg)
        f.write('%s\n'%s)

        traceback.print_tb(sys.exc_info()[2])
        return -1.0

    return m

#------------------------------------------------------------------------------
def pts_per_cycle_function(x, n=1.0, a=10.0, sigma=0.01):
    """
    A function providing a constant numer of points per cycle in the inspiral
    with an increase close to merger.
    The functional form is motivated by comparing against romspline grids for the phase
    and smoothing the data, fitting the shape "by eye".
    Input: x = phi - phi_c
    """
    return n + a*np.exp(sigma*x)

#------------------------------------------------------------------------------
def monotonically_increasing_timeseries(t, v, return_idx=False):
    """
Return the part of a timeseries where the independent and dependent
variables are not monotonically increasing from the start of the data
    """
    idx_t = np.where(np.diff(t) < 0)[0]
    idx_v = np.where(np.diff(v) < 0)[0]

    if len(idx_t) == 0:
        idx_t = len(t)
    else:
        idx_t = idx_t[0]
    if len(idx_v) == 0:
        idx_v = len(v)
    else:
        idx_v = idx_v[0]

    i = min(idx_t, idx_v)

    if return_idx:
        return i
    else:
        return t[:i], v[:i]

#------------------------------------------------------------------------------
def Generate_phase_grid(t, phi):
    """
    Using raw time and phase data, construct a grid that has
    a specific number of points per cycle.
    """
    # Make sure that the phase data is monotonically increasing
    # This is required for the spline interpolation below
    tm, phim = monotonically_increasing_timeseries(t, -phi)

    i = 0
    phi_max = max(phim)
    phi_g = [phim[i]]

    while phi_g[i] < phi_max:
        p = pts_per_cycle_function(phi_g[i] - phi_max, n=1.0, a=10.0, sigma=0.01)
        phi_new = phi_g[i] + 2*np.pi / p
        phi_g.append(phi_new)
        i += 1
    phase_grid = np.array(phi_g[:-1])

    # Remove points where the phase does not increase monotonically
    idx_ok = np.where(np.diff(phim) > 0)

    # Compute time values from spline of original data
    t_grid = spline(phim[idx_ok], tm[idx_ok])(phase_grid)

    # Add points beyond where ceased to be monotonic
    if len(tm) < len(t):
        idx_nm = np.where(t > tm[-1])[0]
        t_grid = np.concatenate([t_grid, t[idx_nm]])
        phase_grid = np.concatenate([phase_grid, -phi[idx_nm]])
    
    return t_grid, phase_grid

#------------------------------------------------------------------------------
def Generate_phase_grid_extrapolate(t, phi):
    """
    Using raw time and phase data, construct a grid that has
    a specific number of points per cycle.
    """
    # Make sure that the phase data is monotonically increasing
    # This is required for the spline interpolation below
    tm, phim = monotonically_increasing_timeseries(t, -phi)

    i = 0
    phi_max = max(phim)
    phi_g = [phim[i]]

    while phi_g[i] < phi_max:
        p = pts_per_cycle_function(phi_g[i] - phi_max, n=1.0, a=10.0, sigma=0.01)
        phi_new = phi_g[i] + 2*np.pi / p
        phi_g.append(phi_new)
        i += 1
    phase_grid = np.array(phi_g[:-1])

    # Remove points where the phase does not increase monotonically
    idx_ok = np.where(np.diff(phim) > 0)

    # Compute time values from spline of original data
    t_grid = spline(phim[idx_ok], tm[idx_ok])(phase_grid)

    # Add points beyond where phase ceased to be monotonic
    if len(tm) < len(t):
        # extrapolate phase
        # cubic extrapolation is more accurate; linear extrapolation would be safer
        phiI = spline(tm[idx_ok], phim[idx_ok], k=3, ext=0)
        idx_nm = np.where(t > tm[-1])[0]
        t_grid = np.concatenate([t_grid, t[idx_nm]])
        phase_grid = np.concatenate([phase_grid, phiI(t[idx_nm])])
    
    return t_grid, phase_grid

#------------------------------------------------------------------------------
def time_grid_for_longest_waveform(Mtot, f_min, outfile, deg=3, abstol=5e-5):
    # Note: This function is no longer used
    # 1) this is very slow for long waveforms (O(day))
    # 2) Ben wants the data to start exactly at f_min.
    #
    # positive aligned spins and equal mass-ratio increase the length of the waveform in time
    # high lambda also makes the wf shorter, so use lambda ~ 0 here.
    
    m1, m2 = Mtot/2.0, Mtot/2.0
    s1z, s2z = +0.9, +0.9 # this is more than we want to cover in spin
    lambda1, lambda2, = 0.1, 0.1
    t, hp, hc = spin_tidal_eob(m1, m2, s1z, s2z, lambda1, lambda2,
                        f_min,
                        distance=1.0, inclination=0.0, delta_t=1.0/16384.0,
                        approximant='TEOBv4', verbose=False)
    # post-process TD data
    h = hp - 1j * hc
    amp = np.abs(h)
    phi = np.unwrap(np.angle(h))
    
    # use a phase romspline for a common grid for amplitude and phase
    spline_phi = romspline.ReducedOrderSpline(t, phi, verbose=True, deg=deg,
                                              tol=abstol, rel=False)

    print 'Generation of phase romspline finished.'
    print 'Size of romspline', spline_phi.size
    print 'Compression factor of romspline', spline_phi.compression
    print 'Resulting spline points in time', spline_phi.X

    phiI = ip.InterpolatedUnivariateSpline(t, phi)
    ampI = ip.InterpolatedUnivariateSpline(t, amp)

    #np.save('./time_grid_debug.npy', [t, hp, hc, amp, phi, spline_phi.X])
    np.save(outfile, spline_phi.X)

#------------------------------------------------------------------------------
def TEOB_process_array_FD(i, M, 
            q, chi1, chi2, lambda1, lambda2,
            f_min, iota, outdir, comm,
            fs, distance, approximant='TEOBv4',
            allow_skip=True, verbose=True):
    '''
    Helper function for workers
    Assumes m1 >= m2
    '''
    args = [i, M, comm.Get_rank(), q, chi1, chi2, lambda1, lambda2, f_min,
            iota, fs, distance]

    config_str = 'TEOB_TD_%d.npy'%i

    if os.path.isfile(outdir+config_str) and allow_skip:
        if verbose:
            print '*** Skipping existing TEOB configuration for parameters:', \
                args
        return
    try:
        print 'Generate wf:', args
        print q
        m1 = M * q/(1.0+q)
        m2 = M * 1.0/(1.0+q)
        t, hp, hc = spin_tidal_eob(m1, m2, chi1, chi2, lambda1, lambda2,
                                f_min,
                                distance=distance, inclination=iota, 
                                delta_t=1.0/fs,
                                approximant=approximant, verbose=False)

        # Compute amplitude and phase and interpolate onto a sparse grid
        h = hp - 1j * hc
        amp = np.abs(h)
        phi = np.unwrap(np.angle(h))
        ampI = ip.InterpolatedUnivariateSpline(t, amp, k=3, ext='zeros')
 
        # Compute phase grid with variable number of points per cycle
        t_grid, phase_grid = Generate_phase_grid_extrapolate(t, phi)

        # use only non-zero amplitude data for constructing spline
        idx = np.where(amp > 0.0)
        ampI = spline(t[idx], amp[idx], k=3, ext='zeros')
        amp_on_grid = ampI(t_grid)

        # Save waveform quantities
        data_save = np.array([t_grid, phase_grid, amp_on_grid])
        np.save(outdir+config_str, data_save)
        # Save raw data for debugging
        config_str_raw = 'TEOB_TD_%d_raw.npy'%i
        data_save_raw = np.array([t, phi, amp])
        np.save(outdir+config_str_raw, data_save_raw)

        if verbose:
            print '*** TEOB_process_array_TD finished for parameters:', args
    except Exception as e:
        print '***********************************************************************'
        print '*** TEOB_process_array_TD failed for parameters:', args
        print '*** Error %s' % e
        print '***********************************************************************'

        f = open('FAILED_process_array_TD_PARAMS.txt', 'a')
        print args
        s = '%.16e'%(args[1])
        for arg in args[2:]:
            if (type(arg) == np.unicode) or (type(arg) == str):
                s += ' '+str(arg)
            elif type(arg) == np.ndarray:
                for a in arg:
                    s += ' %.16e'%(a)
            else:
                s += ' %.16e'%(arg)
        f.write('%s\n'%s)

        traceback.print_tb(sys.exc_info()[2])

#------------------------------------------------------------------------------
def parse_args():

    desc = """
    Evaluate many TEOB waveforms, saving the results to an h5 file.
    All options are specified in a json file."""

    import argparse

    parser = argparse.ArgumentParser(description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--opts', '-o', type=str, required=True,
            help='Path to options json file (see Example.json)')

    args = parser.parse_args()
    opts = sanitize_opts(json.load(open(args.opts)))

    return opts

#------------------------------------------------------------------------------
def process_param_opts(opts):
    opts['subspace_params_1d'] = TS_1d_params(
            opts['min_vals'],
            opts['max_vals'])
    return opts

#------------------------------------------------------------------------------
def sanitize_opts(opts):
    # The string "False" evaluates to True, but the user probably wants False
    if 'random' in opts.keys() and opts['random'] in ["False", "false"]:
        print 'WARNING: converting json string "%s" to python False.'%(
                opts['random'])
        print 'Use the literal false in JSON file, which gets converted to python False'
        opts['random'] = False
    return opts

#------------------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #opts = parse_args()
    verbose = True
    outdir = '/work/mpuer/projects/BNSmatches/SEOBNRv4_ROM_SEOBNRv4_ROM_NRTidal_30Hz_5/'
    tmpdir = outdir + 'tmp/' # Store .npy waveforms here
    if rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
    comm.Barrier()
    os.chdir(outdir)

    # Settings
    Mtot = 2.8
    f_min = 30.0
    f_max = 4096.0
    deltaF = 1.0/128.0

    # generate regular grid in q, spin1z, spin2z, lambda1, lambda2
    qs = np.linspace(1.0, 2.0, 2)
    Lambdas = np.linspace(1.0, 5000.0, 5)
    chis = np.linspace(-0.5, 0.5, 5)

    qi, c1i, c2i, L1i, L2i = np.meshgrid(qs, chis, chis, Lambdas, Lambdas, indexing='ij')
    cfgs = np.vstack([qi.flatten(), c1i.flatten(), c2i.flatten(), L1i.flatten(), L2i.flatten()]).T



    n = len(cfgs)
    if rank == 0: print 'Total number of configurations', n
    m = n / nprocs # # of configurations per core
    if rank == 0: print m, 'configurations per core plus an extra', n - m*nprocs, 'configurations'

    if verbose:
        if rank == 0:
            print '=============================================================='
            for p in cfgs:
                print p
    comm.Barrier()
    
    if rank == 0: 
        print '=============================================================='
        print 'Entering main parallel loop'
    for j in np.arange(m):
        i = j*nprocs + rank
        if (rank == 0) and verbose: print 'Chunk %d out of %d.' %(j,m)
        q, chi1, chi2, lambda1, lambda2 = cfgs[i]
        BNS_tidal_match(i, Mtot*q/(1.0+q), Mtot/(1.0+q), chi1, chi2, lambda1, lambda2,
                    tmpdir, f_min=f_min, f_max=f_max, deltaF=deltaF)

    # Remaining configurations
    if (rank == 0) and verbose: print 'Remaining partial chunk'
    j = m*nprocs
    i = j + rank
    if (i < n):
        q, chi1, chi2, lambda1, lambda2 = cfgs[i]
        BNS_tidal_match(i, Mtot*q/(1.0+q), Mtot/(1.0+q), chi1, chi2, lambda1, lambda2,
                    tmpdir, f_min=f_min, f_max=f_max, deltaF=deltaF)

    if rank == 0: print 'Waiting on other procs...'
    comm.Barrier()
    if rank == 0: print '=============================================================='

    # Load all .npy and save as txt
    if rank == 0:
        print 'Loading npy files for all configurations'

        basename = '%s/match'%(tmpdir)
        matches = []
        for i in tqdm(np.arange(n)):
            f = '%s_%d.npy'%(basename, i)
            try:
                data = np.load(f)
                matches.append(data)
            except Exception as e:
                print 'Loading %s failed. Skipping configuration.' % (f)
        matches = np.array(matches)
        np.savetxt(outdir+'/matches.dat', matches)

        print '=============================================================='
        print 'All Done!'
        print '=============================================================='

if __name__ == "__main__":
    main()
