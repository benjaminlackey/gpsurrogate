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

import os.path
import h5py
import glob

from mpi4py import MPI
import warnings
import json
from multiprocessing import Process, Queue

import romspline

from tqdm import *

# Q: Andrea, do we want 'TEOBv2' or 'TEOBv4'?
# Q: Is the head of TEOBBNS fine?

# Ben
# Q: which distance, inclination do you want me to use?
# Q: which total mass do you want me to use?
# Q: Do you want both hp, hc; since the wf is aligned, there is no need to store both.
# Q: why is tstart needed? shouldn't we just align at the peak of h22?

# Q: Is a common time grid needed? I think not and will return 
    # either a romspline grid for each cfg (but that may be too slow to generate)
    # or data interpolated onto a common romspline grid, but using zeros instead of extrapolation)
    # or the TD amplitude, phase data on a suitable grid
    # or the raw data



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
    # Q: why is this needed? shouldn't we just align at the peak?
    ts = tstart + hp.deltaT*np.arange(hp.data.length)
    
    return ts, hp.data.data, hc.data.data
    #return wave.Waveform.from_hp_hc(ts, hp.data.data, hc.data.data)


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
def example_waveform():
    m1, m2 = 1.4, 1.4
    s1z, s2z = 0.1, 0.1
    lambda1, lambda2, = 2000, 2000
    f_min = 40.0 # O(2min) per waveform
    t, hp, hc = spin_tidal_eob(m1, m2, s1z, s2z, lambda1, lambda2,
                        f_min,
                        distance=1.0, inclination=0.0, delta_t=1.0/16384.0,
                        approximant='TEOBv4', verbose=True)

#------------------------------------------------------------------------------
def TEOB_process_array_TD(i, M, 
            q, chi1, chi2, lambda1, lambda2,
            f_min, iota, outdir, F_out, comm,
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

        m1 = M * q/(1.0+q)
        m2 = M * 1.0/(1.0+q)
        t, hp, hc = spin_tidal_eob(m1, m2, chi1, chi2, lambda1, lambda2,
                                f_min,
                                distance=distance, inclination=iota, 
                                delta_t=1.0/fs,
                                approximant=approximant, verbose=verbose)

        #FIXME: F_out is currently not used; decide what to output
        
        # Save waveform quantities
        data_save = np.array([t, hp, hc])
        np.save(outdir+config_str, data_save)


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
    
    # write function that computes corner cfgs for a specific 5D domain
    # write function that reads list of configs from txt or npy
    # then iterate over these lists and produce wfs (in parallel)

    opts = parse_args()

    # Set parameters from config file
    M = opts['Total_mass_MSUN']
    fs = opts['Sampling_rate_Hz']
    f_min = opts['f_min']
    iota = opts['iota']
    distance = opts['distance_MPC']*1e6*lal.PC_SI
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

    os.chdir(outdir)

    q_min, chi1_min, chi2_min, lambda1_min, lambda2_min = opts['params']['min_vals']
    q_max, chi1_max, chi2_max, lambda1_max, lambda2_max = opts['params']['max_vals']    
    d = domain()
    d.set_q(q_min, q_max)
    d.set_chi1(chi1_min, chi1_max)
    d.set_chi2(chi2_min, chi2_max)
    # FIXME: Andrea: I get nans when calling SimUniversalRelationlambda3TidalVSlambda2Tidal and similar functions with lambda = 0
    d.set_lambda1(lambda1_min, lambda1_max)
    d.set_lambda2(lambda2_min, lambda2_max)
    cfgs = compute_corners_from_domain(d) # 2^5 = 32 corner points

    # FIXME: F_out
    F_out = None


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
        TEOB_process_array_TD(i, M, 
                    q, chi1, chi2, lambda1, lambda2,
                    f_min, iota, tmpdir, F_out, comm,
                    fs, distance, approximant=approximant,
                    allow_skip=True, verbose=True)

    # Remaining configurations
    if (rank == 0) and verbose: print 'Remaining partial chunk'
    j = m*nprocs
    i = j + rank
    if (i < n):
        q, chi1, chi2, lambda1, lambda2 = cfgs[i]
        TEOB_process_array_TD(i, M, 
                    q, chi1, chi2, lambda1, lambda2,
                    f_min, iota, tmpdir, F_out, comm,
                    fs, distance, approximant=approximant,
                    allow_skip=True, verbose=True)

    if rank == 0: print 'Waiting on other procs...'
    comm.Barrier()
    if rank == 0: print '=============================================================='

    # FIXME: add consolidation of data step
    
    print '=============================================================='
    print 'All Done!'
    print '=============================================================='

if __name__ == "__main__":
    main()
