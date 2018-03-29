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

from tqdm import *

# MP 08 / 2017

def FD_waveform_generate(m1, m2, chi1, chi2, lambda1, lambda2, 
    f_min=20.0, f_max=2048.0, deltaF=0, 
    approximant='SEOBNRv4T',
    distance=1.0, inclination=0.0):

    phiRef, fRef = 0.0, f_min

    m1SI, m2SI, chi1, chi2, lambda1, lambda2, nk_max = m1*lal.MSUN_SI, m2*lal.MSUN_SI, chi1, chi2, lambda1, lambda2, -1

    longAscNodes, eccentricity, meanPerAno = 0,0,0

    LALpars = lal.CreateDict()
    LS.SimInspiralWaveformParamsInsertTidalLambda1(LALpars, lambda1)
    LS.SimInspiralWaveformParamsInsertTidalLambda2(LALpars, lambda2)

    # Nyquist frequency is set by fHigh
    # Can set deltaF = 0 to figure out required frequency spacing; the chosen deltaF depends on fLow
    
    # Documentation from LALSimInspiral.c
    #
    #  * This routine can generate TD approximants and transform them into the frequency domain.
    #  * Waveforms are generated beginning at a slightly lower starting frequency and tapers
    #  * are put in this early region so that the waveform smoothly turns on.
    #  *
    #  * If an FD approximant is used, this routine applies tapers in the frequency domain
    #  * between the slightly-lower frequency and the requested f_min.  Also, the phase of the
    #  * waveform is adjusted to introduce a time shift.  This time shift should allow the
    #  * resulting waveform to be Fourier transformed into the time domain without wrapping
    #  * the end of the waveform to the beginning.
    #  *
    #  * This routine assumes that f_max is the Nyquist frequency of a corresponding time-domain
    #  * waveform, so that deltaT = 0.5 / f_max.  If deltaF is set to 0 then this routine computes
    #  * a deltaF that is small enough to represent the Fourier transform of a time-domain waveform.
    #  * If deltaF is specified but f_max / deltaF is not a power of 2, and the waveform approximant
    #  * is a time-domain approximant, then f_max is increased so that f_max / deltaF is the next
    #  * power of 2.  (If the user wishes to discard the extra high frequency content, this must
    #  * be done separately.)
    
    lal_approx = LS.GetApproximantFromString(approximant)
    hp, hc = LS.SimInspiralFD(m1SI, m2SI,
                     0.0, 0.0, chi1,
                     0.0, 0.0, chi2,
                     distance, inclination, phiRef, 
                     longAscNodes, eccentricity, meanPerAno, 
                     deltaF,
                     f_min, f_max, fRef,
                     LALpars,
                     lal_approx)

    fHz = np.arange(hp.data.length)*hp.deltaF
    h = hp.data.data + 1j*hc.data.data

    return fHz, h

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
def BNS_tidal_match(i, m1, m2, chi1, chi2, lambda1, lambda2, tmpdir,
                    f_min=40.0, f_max=2048.0, deltaF=1/128.0, allow_skip=True, zpf=2):
    args = i, m1, m2, chi1, chi2, lambda1, lambda2, f_min, f_max, deltaF
    print '*** Starting BNS_tidal_match for parameters:', args
  
    if os.path.isfile(tmpdir+'match_'+str(i)+'.npy') and allow_skip:
        print '*** Skipping existing configuration for parameters:', args
        return
 
    distance = 1.0e6
    inclination = 0.0

    try:
        f1, h1 = FD_waveform_generate(m1, m2, chi1, chi2, lambda1, lambda2,
                        f_min=f_min, f_max=f_max, deltaF=deltaF,
                        distance=distance, inclination=inclination,
                        approximant='SEOBNRv4T')

        f2, h2 = FD_waveform_generate(m1, m2, chi1, chi2, lambda1, lambda2,
                        f_min=f_min, f_max=f_max, deltaF=deltaF,
                        distance=distance, inclination=inclination,
                        approximant='SEOBNRv4T_surrogate')
        assert np.unique(np.diff(f1)), np.unique(np.diff(f2))
        if deltaF == 0:
            deltaF = np.unique(np.diff(f1))

        # Downsample the data for the match computation (saw difference in 7th digit after the comma)
        s = 10 # will give ~ 0.1 Hz for deltaF=1.0/128.0
        m = match(h1[::s], h2[::s], 
              LS.SimNoisePSDaLIGOZeroDetHighPower, 
              deltaF*s, zpf=zpf, verbose=False)

        np.save(tmpdir+'match_'+str(i)+'.npy', np.array([m1, m2, chi1, chi2, lambda1, lambda2, m]))
        np.save(tmpdir+'wfs_'+str(i)+'.npy', np.array([f1, h1, h2]))

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
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    verbose = True
    outdir = '/work/mpuer/projects/BNSmatches/30Hz_5/'
    outdir = './'
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
    f_min = 20.0
    f_max = 8192.0
    deltaF = 1.0/128.0

    # generate regular grid in q, spin1z, spin2z, lambda1, lambda2
    n_q = 5
    n_L = 5
    n_chi = 5
    qs = np.linspace(1.0, 3.0, n_q)
    Lambdas = np.linspace(1.0, 5000.0, n_L)
    chis = np.linspace(-0.5, 0.5, n_chi)

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
        np.savetxt(outdir+'/matches_f_min_%f_f_max_%f_df_%f.dat'%(f_min, f_max, deltaF), matches)

        print '=============================================================='
        print 'All Done!'
        print '=============================================================='

if __name__ == "__main__":
    main()
