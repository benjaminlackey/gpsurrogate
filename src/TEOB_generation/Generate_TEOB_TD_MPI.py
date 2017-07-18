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

# MP 06-07, 2017

#------------------------------------------------------------------------------
def spin_tidal_eob(m1, m2, s1z, s2z, lambda1, lambda2,
                    f_min, 
                    delta_t=1.0/16384.0, distance=1.0, inclination=0.0,
                    approximant='TEOBv4', verbose=True):
    """EOB waveform with aligned spin and tidal interactions. 
    l=3 tidal interaction and l=2,3 f-mode calculated with universal relations.
    
    Parameters
    ----------
    approximant : 'TEOBv2' or 'TEOBv4'
        Based on the inspiral model given by 'SEOBNRv2' or 'SEOBNRv4'.
    
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
    if (approximant not in ['TEOBv2', 'TEOBv4']):
        raise Exception, "Approximant must be 'TEOBv2' or 'TEOBv4'."
    lal_approx = LS.GetApproximantFromString(approximant)
    
    
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

    # print 'Terms calculated from universal relations'
    # print lambda31_ur, lambda32_ur
    # print omega21_ur, omega22_ur
    # print omega31_ur, omega32_ur

    # Insert matter parameters
    lal_params = lal.CreateDict()
    LS.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda1)
    LS.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, lambda2)
    LS.SimInspiralWaveformParamsInsertTidalOctupolarLambda1(lal_params, lambda31_ur)
    LS.SimInspiralWaveformParamsInsertTidalOctupolarLambda2(lal_params, lambda32_ur)
    LS.SimInspiralWaveformParamsInsertTidalQuadrupolarFMode1(lal_params, omega21_ur)
    LS.SimInspiralWaveformParamsInsertTidalQuadrupolarFMode2(lal_params, omega22_ur)
    LS.SimInspiralWaveformParamsInsertTidalOctupolarFMode1(lal_params, omega31_ur)
    LS.SimInspiralWaveformParamsInsertTidalOctupolarFMode2(lal_params, omega32_ur)
    
    if verbose:
        ap = LS.GetStringFromApproximant(lal_approx)
        L2A = LS.SimInspiralWaveformParamsLookupTidalLambda1(lal_params)
        L2B = LS.SimInspiralWaveformParamsLookupTidalLambda2(lal_params)
        L3A = LS.SimInspiralWaveformParamsLookupTidalOctupolarLambda1(lal_params)
        L3B = LS.SimInspiralWaveformParamsLookupTidalOctupolarLambda2(lal_params)
        w2A = LS.SimInspiralWaveformParamsLookupTidalQuadrupolarFMode1(lal_params)
        w2B = LS.SimInspiralWaveformParamsLookupTidalQuadrupolarFMode2(lal_params)
        w3A = LS.SimInspiralWaveformParamsLookupTidalOctupolarFMode1(lal_params)
        w3B = LS.SimInspiralWaveformParamsLookupTidalOctupolarFMode2(lal_params)
        print 'Approximant: '+str(ap)
        print 'm1={:.2f}, m2={:.2f}'.format(m1, m2)
        print 's1z={:.2f}, s2z={:.2f}'.format(s1z, s2z)
        print 'delta_t={:.6f}, 1/delta_t={:.5}, f_min={:.2f}'.format(delta_t, 1./delta_t, f_min)
        print 'L2A, L2B, L3A, L3B, w2A, w2B, w3A, w3B:'
        print '{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(L2A, L2B, L3A, L3B, w2A, w2B, w3A, w3B)
        sys.stdout.flush()
    
    # Evaluate waveform
    hp, hc = LS.SimInspiralChooseTDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 
        s1x, s1y, s1z, s2x, s2y, s2z, 
        distance*lal.PC_SI,
        inclination, phiRef, longAscNodes, eccentricity, meanPerAno,
        delta_t, f_min, f_ref, lal_params, lal_approx)
    
    # Extract time array from lalsimulation's structures
    tstart = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds*1.0e-9 
    ts = tstart + hp.deltaT*np.arange(hp.data.length)
    
    return ts, hp.data.data, hc.data.data

#------------------------------------------------------------------------------
class domain:
    """
    Store parameter ranges for 5D model domain
    """
    def __init__(self):
        self.q_min = None
        self.q_max = None
        self.chi1_min = None
        self.chi1_max = None
        self.chi2_min = None
        self.chi2_max = None
        self.lambda1_min = None
        self.lambda1_max = None
        self.lambda2_min = None
        self.lambda2_max = None
    def set_q(self, q_min, q_max):
        self.q_min = q_min
        self.q_max = q_max
    def set_chi1(self, chi1_min, chi1_max):
        self.chi1_min = chi1_min
        self.chi1_max = chi1_max
    def set_chi2(self, chi2_min, chi2_max):
        self.chi2_min = chi2_min
        self.chi2_max = chi2_max
    def set_lambda1(self, lambda1_min, lambda1_max):
        self.lambda1_min = lambda1_min
        self.lambda1_max = lambda1_max
    def set_lambda2(self, lambda2_min, lambda2_max):
        self.lambda2_min = lambda2_min
        self.lambda2_max = lambda2_max
    def get_q(self):
        return (self.q_min, self.q_max)
    def get_chi1(self):
        return (self.chi1_min, self.chi1_max)
    def get_chi2(self):
        return (self.chi2_min, self.chi2_max)
    def get_lambda1(self):
        return (self.lambda1_min, self.lambda1_max)
    def get_lambda2(self):
        return (self.lambda2_min, self.lambda2_max)

#------------------------------------------------------------------------------
def compute_corners_from_domain(d):
    return [(q, chi1, chi2, lambda1, lambda2)
                                                for q in d.get_q() 
                                                for chi1 in d.get_chi1()
                                                for chi2 in d.get_chi2()
                                                for lambda1 in d.get_lambda1()
                                                for lambda2 in d.get_lambda2()]

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
def example_waveform():
    m1, m2 = 3.0, 1.0
    s1z, s2z = 0.1, 0.1
    lambda1, lambda2, = 2000, 2000
    f_min = 40.0 # O(2min) per waveform
    t, hp, hc = spin_tidal_eob(m1, m2, s1z, s2z, lambda1, lambda2,
                        f_min,
                        distance=1.0, inclination=0.0, delta_t=1.0/16384.0,
                        approximant='TEOBv4', verbose=True)
    return t, hp, hc

#------------------------------------------------------------------------------
def TEOB_process_array_TD(i, M, 
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

    opts = parse_args()

    # Set parameters from config file
    M = opts['Total_mass_MSUN']
    fs = opts['Sampling_rate_Hz']
    f_min = opts['f_min']
    iota = opts['iota']
    distance = opts['distance_MPC']*1e6
    approximant = str(opts['approximant'])
    
    try:
        verbose = opts['verbose']
    except:
        verbose = True
        pass

    outdir = opts['outdir']
    tmpdir = outdir + '/tmp/' # Store .npy waveforms here
    if rank == 0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
    comm.Barrier()
    os.chdir(outdir)

    if opts['cfgfile']:
        cfgfile = opts['cfgfile']
        print 'Loading configurations from textfile', cfgfile
        cfgs = np.loadtxt(cfgfile)
        if cfgs.shape[1] != 5:
            raise ValueError("Expected configurations in file %s to contain 5 columns "
            +"(q, spin1z, spin2z, lambda1, lambda2)."%cfgfile)
        if np.min(cfgs.T[0]) > 1.0:
            raise ValueError("Expected in file %s mass-ratio <= 1"%cfgfile)
    elif opts['corners']:
        print 'Computing corners of specified 5D domain'
        q_min, chi1_min, chi2_min, lambda1_min, lambda2_min = opts['params']['min_vals']
        q_max, chi1_max, chi2_max, lambda1_max, lambda2_max = opts['params']['max_vals']    
        d = domain()
        d.set_q(q_min, q_max)
        d.set_chi1(chi1_min, chi1_max)
        d.set_chi2(chi2_min, chi2_max)
        d.set_lambda1(lambda1_min, lambda1_max)
        d.set_lambda2(lambda2_min, lambda2_max)
        cfgs = compute_corners_from_domain(d) # 2^5 = 32 corner points
    else:
        print 'Using adhoc 1-parameter family for tests of smoothness'
        q = 0.6
        spin1z_arr = np.linspace(-0.7, 0.7, 51)
        spin2z = 0.2
        lambda1 = 2000.0
        lambda2 = 1000.0
        cfgs = [(q, spin1z, spin2z, lambda1, lambda2) for spin1z in spin1z_arr]

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
        # Note: We expect q <= 1. TEOB_process_array_TD() uses the opposite convention.
        TEOB_process_array_TD(i, M, 
                    1.0/q, chi1, chi2, lambda1, lambda2,
                    f_min, iota, tmpdir, comm,
                    fs, distance, approximant=approximant,
                    allow_skip=True, verbose=True)

    # Remaining configurations
    if (rank == 0) and verbose: print 'Remaining partial chunk'
    j = m*nprocs
    i = j + rank
    if (i < n):
        q, chi1, chi2, lambda1, lambda2 = cfgs[i]
        TEOB_process_array_TD(i, M, 
                    1.0/q, chi1, chi2, lambda1, lambda2,
                    f_min, iota, tmpdir, comm,
                    fs, distance, approximant=approximant,
                    allow_skip=True, verbose=True)

    if rank == 0: print 'Waiting on other procs...'
    comm.Barrier()
    if rank == 0: print '=============================================================='

    # Load all .npy and save as hdf5
    if rank == 0:
        print 'Loading npy files for all configurations'

        # FIXME: add git branch and hash
        # how to find build directory? then can use git rev-parse HEAD
        generation_str = \
        '''
        approximant = %s
        M = %g
        fs = %g
        iota = %g
        f_min = %g
        distance = %g

        Using lalsimulation version %s
        ''' % (approximant, M, fs, iota, f_min, distance, LS.LALSIMULATION_VERSION)

        print 'Outputting interpolated data as HDF5'
        outfname = 'TEOB_dataI.hdf5'
        fh5 = h5py.File(outfname, 'w')
        fh5.create_dataset('configurations', data=cfgs)
        fh5.create_dataset('configurations_keys', data=['q', 'chi1', 'chi2', 'lambda1', 'lambda2'])
        fh5.attrs['Description'] = np.string_('TEOB TD data')
        fh5.attrs['GenerationSettings'] = generation_str

        data_keys_name = ['t', 'phi', 'amp']
        basename = '%s/TEOB_TD'%(tmpdir)
        fh5.create_dataset('data_keys_name', data=data_keys_name)
        for i in tqdm(np.arange(n)):
            if verbose:
                print 'Loading data for waveform %d of %d with index %d.' %(i,n,cfgs[i])
            f = '%s_%d.npy'%(basename, i)
            try:
                data = np.load(f)
                grp = fh5.create_group('h_'+str(i))
                for j in range(len(data_keys_name)):
                # create one data set for each quantity
                # each data set includes data as a function of
                # time and physical parameter(s)
                    if np.isnan(data[j]).any():
                        print '*'*80
                        print '*** Warning: dataset %d for quantity %s contains one or more nans! ***' %(i, data_keys_name[j])
                        print data[j]
                        print '*'*80
                    grp.create_dataset(data_keys_name[j], data=data[j])
            except Exception as e:
                print 'Loading %s failed. Skipping configuration.' % (f)

        fh5.close()
    
        print '=============================================================='
        print 'All Done!'
        print '=============================================================='

if __name__ == "__main__":
    main()
