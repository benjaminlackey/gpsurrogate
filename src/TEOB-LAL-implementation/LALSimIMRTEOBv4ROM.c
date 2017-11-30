/*
 *  Copyright (C) 2017 Michael Puerrer, Ben Lackey
 *  Reduced Order Model for TEOBv4
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#ifdef __GNUC__
#define UNUSED __attribute__ ((unused))
#else
#define UNUSED
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdbool.h>
#include <alloca.h>
#include <string.h>
#include <libgen.h>
#include <assert.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <lal/Units.h>
#include <lal/SeqFactories.h>
#include <lal/LALConstants.h>
#include <lal/XLALError.h>
#include <lal/FrequencySeries.h>
#include <lal/Date.h>
#include <lal/StringInput.h>
#include <lal/Sequence.h>
#include <lal/LALStdio.h>
#include <lal/FileIO.h>


#ifdef LAL_HDF5_ENABLED
#include <lal/H5FileIO.h>
// FIXME: change these to our datafile and version number 1.0.0
//static const char ROMDataHDF5[] = "SEOBNRv4ROM_v2.0.hdf5";
//static const char ROMDataHDF5[] = "/Users/mpuer/Documents/gpsurrogate/src/TEOB-LAL-implementation/TEOBv4_surrogate.hdf5";
// FIXME: missing attributes in HDF5 file: Email, Description (rather than description), version_major, version_minor, version_micro
static const char ROMDataHDF5[] = "TEOBv4_surrogate.hdf5";
// FIXME: uncomment after adding attributes
// static const INT4 ROMDataHDF5_VERSION_MAJOR = 1;
// static const INT4 ROMDataHDF5_VERSION_MINOR = 0;
// static const INT4 ROMDataHDF5_VERSION_MICRO = 0;
#endif

#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>

#include "LALSimIMREOBNRv2.h"
#include "LALSimInspiralPNCoefficients.c"
#include "LALSimIMRSEOBNRROMUtilities.c"

#include <lal/LALConfig.h>
#ifdef LAL_PTHREAD_LOCK
#include <pthread.h>
#endif


// TODO:
// * add waveform approximant and glueing code in LALSimInspiral
// * remove all debugging code: fprintf statements
// * add more checks and tests
// * take care of all FIXMEs


#ifdef LAL_PTHREAD_LOCK
static pthread_once_t TEOBv4ROM_is_initialized = PTHREAD_ONCE_INIT;
#endif

/*************** type definitions ******************/

// Constants if needed
//static const INT4 n_pars = 5; // dimensionality of the parameter space
//static const INT4 n_hyp = 7; // number of hyperparameters in GP kernel
//static const INT4 n_train = 159; // number of training points for the GPR
//static const INT4 n_EI_nodes = 20; // number of empirical interpolant nodes (or dimension of EI basis)
//static const INT4 n_freqs = 10000; // number of frequency points in bases
// n_EI_nodes and n_freqs are the same number for amplitude and phase


struct tagTEOBv4ROMdataDS_submodel
{
  // /B_amp                   Dataset {20, 10000}
  // /B_phi                   Dataset {20, 10000}
  // /EI_nodes_amp            Dataset {20}
  // /EI_nodes_phi            Dataset {20}
  // /hyp_amp                 Dataset {20, 7}
  // /hyp_phi                 Dataset {20, 7}
  // /kinv_dot_y_amp          Dataset {20, 159}
  // /kinv_dot_y_phi          Dataset {20, 159}
  // /mf                      Dataset {10000}
  // /x_train                 Dataset {159, 5}

  gsl_matrix *hyp_amp;         // GP hyperparameters log amplitude
  gsl_matrix *hyp_phi;         // GP hyperparameters for dephasing
  gsl_matrix *kinv_dot_y_amp;  // kinv_dot_y for log amplitude
  gsl_matrix *kinv_dot_y_phi;  // kinv_dot_y for dephasing
  gsl_matrix *B_amp;           // Reduced basis for log amplitude
  gsl_matrix *B_phi;           // Reduced basis for dephasing
  gsl_matrix *x_train;         // Training points
  gsl_vector *mf_amp;          // location of spline nodes for log amplitude
  gsl_vector *mf_phi;          // location of spline nodes for dephasing

  // 5D parameter space bounds of surrogate
  double q_bounds[2];          // [q_min, q_max]
  double chi1_bounds[2];       // [chi1_min, chi1_max]
  double chi2_bounds[2];       // [chi2_min, chi2_max]
  double lambda1_bounds[2];    // [lambda1_min, lambda1_max]
  double lambda2_bounds[2];    // [lambda2_min, lambda2_max]
};
typedef struct tagTEOBv4ROMdataDS_submodel TEOBv4ROMdataDS_submodel;

struct tagTEOBv4ROMdataDS
{
  UINT4 setup;
  TEOBv4ROMdataDS_submodel* sub1;
};
typedef struct tagTEOBv4ROMdataDS TEOBv4ROMdataDS;

static TEOBv4ROMdataDS __lalsim_TEOBv4ROMDS_data;

typedef int (*load_dataPtr)(const char*, gsl_vector *, gsl_vector *, gsl_matrix *, gsl_matrix *, gsl_vector *);

/**************** Internal functions **********************/

static gsl_vector *gsl_vector_prepend_value(gsl_vector *v, double value);

double kernel(
  gsl_vector *x1,          // array with shape ndim
  gsl_vector *x2,          // array with shape ndim
  gsl_vector *hyperparams  // Hyperparameters
);

double gp_predict(
  gsl_vector *xst,          // Point x_* where you want to evaluate the function.
  gsl_vector *hyperparams,  // Hyperparameters for the GPR kernel.
  gsl_matrix *x_train,      // Training set points.
  gsl_vector *Kinv_dot_y    // The interpolating weights at each training set point.
);


UNUSED static void TEOBv4ROM_Init_LALDATA(void);
UNUSED static int TEOBv4ROM_Init(const char dir[]);
UNUSED static bool TEOBv4ROM_IsSetup(void);

UNUSED static int TEOBv4ROMdataDS_Init(TEOBv4ROMdataDS *romdata, const char dir[]);
UNUSED static void TEOBv4ROMdataDS_Cleanup(TEOBv4ROMdataDS *romdata);

static int GPR_evaluation_5D(
  double q,                      // Input: q-value (q >= 1)
  double chi1,                   // Input: chi1-value
  double chi2,                   // Input: chi2-value
  double lambda1,                // Input: lambda1-value
  double lambda2,                // Input: lambda2-value
  gsl_matrix *hyp_amp,           // Input: GPR hyperparameters for log amplitude
  gsl_matrix *hyp_phi,           // Input: GPR hyperparameters for dephasing
  gsl_matrix *kinv_dot_y_amp,    // Input: kinv_dot_y for log amplitude
  gsl_matrix *kinv_dot_y_phi,    // Input: kinv_dot_y for dephasing
  gsl_matrix *x_train,           // Input: GPR training points
  gsl_vector *amp_at_EI_nodes,   // Output: log amplitude at EI nodes (preallocated)
  gsl_vector *phi_at_EI_nodes    // Output: dephasing at EI nodes     (preallocated)
);

UNUSED static int TEOBv4ROMdataDS_Init_submodel(
  UNUSED TEOBv4ROMdataDS_submodel **submodel,
  UNUSED const char dir[],
  UNUSED const char grp_name[]
);

UNUSED static void TEOBv4ROMdataDS_Cleanup_submodel(TEOBv4ROMdataDS_submodel *submodel);

/**
 * Core function for computing the ROM waveform.
 * Interpolate projection coefficient data and evaluate coefficients at desired (q, chi).
 * Construct 1D splines for amplitude and phase.
 * Compute strain waveform from amplitude and phase.
*/
UNUSED static int TEOBv4ROMCore(
  COMPLEX16FrequencySeries **hptilde,
  COMPLEX16FrequencySeries **hctilde,
  double phiRef,
  double fRef,
  double distance,
  double inclination,
  double Mtot_sec,
  double eta,
  double chi1,
  double chi2,
  double lambda1,
  double lambda2,
  const REAL8Sequence *freqs, /* Frequency points at which to evaluate the waveform (Hz) */
  double deltaF
  /* If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
);

static size_t NextPow2(const size_t n);

static int TaylorF2Phasing(
  double Mtot,         // Total mass in solar masses
  double q,            // Mass-ration m1/m2 >= 1
  double chi1,         // Dimensionless aligned spin of body 1
  double chi2,         // Dimensionless aligned spin of body 2
  double lambda1,      // Tidal deformability of body 1
  double lambda2,      // Tidal deformability of body 2
  double dquadmon1,    // Self-spin deformation of body 1
  double dquadmon2,    // Self-spin deformation of body 2
  gsl_vector *Mfs,     // Input geometric frequencies
  gsl_vector **PNphase // Output: TaylorF2 phase at frequencies Mfs
);

static int TaylorF2Amplitude1PN(
  double eta,        // Symmetric mass-ratio
  gsl_vector *Mfs,   // Input geometric frequencies
  gsl_vector **PNamp // Output: TaylorF2 amplitude at frequencies Mfs
);

/********************* Definitions begin here ********************/

static gsl_vector *gsl_vector_prepend_value(gsl_vector *v, double value) {
// Helper function to prepend a value to a gsl_vector
// Returns the augmented gsl_vector
// Deallocates the input gsl_vector
  int n = v->size;
  gsl_vector *vout = gsl_vector_alloc(n+1);

  gsl_vector_set(vout, 0, value);
  for (int i=1; i<=n; i++)
    gsl_vector_set(vout, i, gsl_vector_get(v, i-1));
  gsl_vector_free(v);

  return vout;
}

double kernel(
  gsl_vector *x1,          // parameter space point 1
  gsl_vector *x2,          // parameter space point 2
  gsl_vector *hyperparams  // GPR Hyperparameters
)
{
  // Matern covariance function for n-dimensional data.
  //
  // Parameters
  // ----------
  // x1 : array with shape ndim
  // x2 : array with shape ndim
  // hyperparams : array with shape ndim+2 [sigma_f, ls0, ls1, ..., sigma_n]
  //     sigma_f : Approximately the range (ymax-ymin) of values that the data takes.
  //         sigma_f^2 called the signal variance.
  //     sigma_n : Noise term. The uncertainty in the y values of the data.
  //     lsi : Length scales for the variation in dimension i.
  //
  // Returns
  // -------
  // covariance : double

  double sigma_f = gsl_vector_get(hyperparams, 0);
  double sigma_n = gsl_vector_get(hyperparams, hyperparams->size-1);
  gsl_vector ls = gsl_vector_subvector(hyperparams, 1, hyperparams->size-2).vector;

  // fprintf(stderr, "** kernel **\n");
  // fprintf(stderr, "x1 = [%g, %g, %g, %g, %g]\n", gsl_vector_get(x1, 0), gsl_vector_get(x1, 1), gsl_vector_get(x1, 2), gsl_vector_get(x1, 3), gsl_vector_get(x1, 4));
  // fprintf(stderr, "x2 = [%g, %g, %g, %g, %g]\n", gsl_vector_get(x2, 0), gsl_vector_get(x2, 1), gsl_vector_get(x2, 2), gsl_vector_get(x2, 3), gsl_vector_get(x2, 4));
  // fprintf(stderr, "ls = [%g, %g, %g, %g, %g]\n", gsl_vector_get(&ls, 0), gsl_vector_get(&ls, 1), gsl_vector_get(&ls, 2), gsl_vector_get(&ls, 3), gsl_vector_get(&ls, 4));

  XLAL_CHECK((x1->size == x2->size) && (x1->size == ls.size), XLAL_EDIMS,
  "kernel(): dimensions of vectors x1, x2 and ls: %zu, %zu, %zu have to be consistent.\n",
  x1->size, x2->size, ls.size);

  // Noise nugget for diagonal elements
  double nugget = 0.0;
  if (gsl_vector_equal(x1, x2))
    nugget = sigma_n*sigma_n;

  gsl_vector *tmp = gsl_vector_alloc(x1->size);
  gsl_vector_memcpy(tmp, x1);
  gsl_vector_sub(tmp, x2);
  gsl_vector_div(tmp, &ls); // (x1 - x2) / ls
  double r = gsl_blas_dnrm2(tmp);
  gsl_vector_free(tmp);

  // nu = 5/2 Matern covariance
  double matern = (1.0 + sqrt(5.0)*r + 5.0*r*r/3.0) * exp(-sqrt(5.0)*r);

  // Full covariance
  // Include the nugget to agree with scikit-learn when the points x1, x2 are exactly the same.
  return sigma_f*sigma_f * matern + nugget;
}

double gp_predict(
  gsl_vector *xst,          // Point x_* where you want to evaluate the function.
  gsl_vector *hyperparams,  // Hyperparameters for the GPR kernel.
  gsl_matrix *x_train,      // Training set points.
  gsl_vector *Kinv_dot_y    // The interpolating weights at each training set point.
)
{
  // Interpolate the function at the point xst using Gaussian process regression.
  //
  // Parameters
  // ----------
  // xst : array of shape ndim.
  //     Point x_* where you want to evaluate the function.
  // hyperparams : array with shape ndim+2 [sigma_f, ls0, ls1, ..., sigma_n].
  //     Hyperparameters for the GPR kernel.
  // x_train : array of shape (n_train, ndim).
  //     Training set points.
  // Kinv_dot_y : array of shape n_train.
  //     The interpolating weights at each training set point.
  //
  // Returns
  // -------
  // yst : double
  //     Interpolated value at the point xst.

  // Evaluate vector K_*
  int n = x_train->size1;
  gsl_vector *Kst = gsl_vector_alloc(n);
  for (int i=0; i < n; i++) {
    gsl_vector x = gsl_matrix_const_row(x_train, i).vector;
    double ker = kernel(xst, &x, hyperparams);
    gsl_vector_set(Kst, i, ker);
  }

  // Evaluate y_*
  double res = 0;
  gsl_blas_ddot(Kst, Kinv_dot_y, &res);
  gsl_vector_free(Kst);
  return res;
}

/** Setup TEOBv4ROM model using data files installed in dir
 */
static int TEOBv4ROM_Init(const char dir[]) {
  if(__lalsim_TEOBv4ROMDS_data.setup) {
    XLALPrintError("Error: TEOBv4ROM data was already set up!");
    XLAL_ERROR(XLAL_EFAILED);
  }
  TEOBv4ROMdataDS_Init(&__lalsim_TEOBv4ROMDS_data, dir);

  if(__lalsim_TEOBv4ROMDS_data.setup) {
    return(XLAL_SUCCESS);
  }
  else {
    return(XLAL_EFAILED);
  }
}

/** Helper function to check if the TEOBv4ROM model has been initialised */
static bool TEOBv4ROM_IsSetup(void) {
  if(__lalsim_TEOBv4ROMDS_data.setup)
    return true;
  else
    return false;
}

// Compute amplitude and phase at empirical interpolant nodes from GPRs.
// This entails interpolation over the 5D parameter space (q, chi1, chi2, lambda1, lambda2).
static int GPR_evaluation_5D(
  double q,                      // Input: q-value (q >= 1)
  double chi1,                   // Input: chi1-value
  double chi2,                   // Input: chi2-value
  double lambda1,                // Input: lambda1-value
  double lambda2,                // Input: lambda2-value
  gsl_matrix *hyp_amp,           // Input: GPR hyperparameters for log amplitude
  gsl_matrix *hyp_phi,           // Input: GPR hyperparameters for dephasing
  gsl_matrix *kinv_dot_y_amp,    // Input: kinv_dot_y for log amplitude
  gsl_matrix *kinv_dot_y_phi,    // Input: kinv_dot_y for dephasing
  gsl_matrix *x_train,           // Input: GPR training points
  gsl_vector *amp_at_nodes,      // Output: log amplitude at frequency nodes (preallocated)
  gsl_vector *phi_at_nodes       // Output: dephasing at frequency nodes (preallocated)
)
{
  // assemble evaluation point
  gsl_vector *xst = gsl_vector_alloc(5);
  double q_inv = 1.0/q;
  gsl_vector_set(xst, 0, q_inv);
  gsl_vector_set(xst, 1, chi1);
  gsl_vector_set(xst, 2, chi2);
  gsl_vector_set(xst, 3, lambda1);
  gsl_vector_set(xst, 4, lambda2);

  // FIXME: find number of spline nodes
  // FIXME: check that amp_at_nodes, phi_at_nodes are preallocated and have the correct size (number of spline nodes)

  // evaluate GPR for amplitude spline nodes
  fprintf(stderr, "\n\n");
  for (size_t i=0; i<amp_at_nodes->size; i++) {
    gsl_vector hyp_amp_i = gsl_matrix_const_row(hyp_amp, i).vector;
    gsl_vector kinv_dot_y_amp_i = gsl_matrix_const_row(kinv_dot_y_amp, i).vector;
    double pred = gp_predict(xst, &hyp_amp_i, x_train, &kinv_dot_y_amp_i);
    fprintf(stderr, "pred_amp(%zu) = %g\n", i, pred);
    gsl_vector_set(amp_at_nodes, i, pred);
  }
  fprintf(stderr, "\n\n");

  // evaluate GPR for phase spline nodes
  fprintf(stderr, "\n\n");
  for (size_t i=0; i<phi_at_nodes->size; i++) {
    gsl_vector hyp_phi_i = gsl_matrix_const_row(hyp_phi, i).vector;
    gsl_vector kinv_dot_y_phi_i = gsl_matrix_const_row(kinv_dot_y_phi, i).vector;
    double pred = gp_predict(xst, &hyp_phi_i, x_train, &kinv_dot_y_phi_i);
    fprintf(stderr, "pred_phi(%zu) = %g\n", i, pred);
    gsl_vector_set(phi_at_nodes, i, pred);
  }
  fprintf(stderr, "\n\n");

  return XLAL_SUCCESS;
}

/* Set up a new ROM submodel, using data contained in dir */
UNUSED static int TEOBv4ROMdataDS_Init_submodel(
  TEOBv4ROMdataDS_submodel **submodel,
  UNUSED const char dir[],
  UNUSED const char grp_name[]
) {
  int ret = XLAL_FAILURE;

  if(!submodel) exit(1);
  /* Create storage for submodel structures */
  if (!*submodel)
    *submodel = XLALCalloc(1,sizeof(TEOBv4ROMdataDS_submodel));
  else
    TEOBv4ROMdataDS_Cleanup_submodel(*submodel);

#ifdef LAL_HDF5_ENABLED
  size_t size = strlen(dir) + strlen(ROMDataHDF5) + 2;
  char *path = XLALMalloc(size);
  snprintf(path, size, "%s/%s", dir, ROMDataHDF5);

  LALH5File *file = XLALH5FileOpen(path, "r");
  //LALH5File *file = XLALH5FileOpen("/Users/mpuer/Documents/gpsurrogate/src/TEOB-LAL-implementation/TEOBv4_surrogate.hdf5", "r");
  LALH5File *root = XLALH5GroupOpen(file, "/");

  //////////////////////////////////////////////////////////////////////////////
  // load everything we need
  // GP hyperparameters
  ReadHDF5RealMatrixDataset(root, "hyp_amp", & (*submodel)->hyp_amp);
  ReadHDF5RealMatrixDataset(root, "hyp_phi", & (*submodel)->hyp_phi);

  // kinv_dot_y
  ReadHDF5RealMatrixDataset(root, "kinv_dot_y_amp", & (*submodel)->kinv_dot_y_amp);
  ReadHDF5RealMatrixDataset(root, "kinv_dot_y_phi", & (*submodel)->kinv_dot_y_phi);

  // Training points
  ReadHDF5RealMatrixDataset(root, "x_train", & (*submodel)->x_train);

  // Frequency grid
  ReadHDF5RealVectorDataset(root, "spline_nodes_amp", & (*submodel)->mf_amp);
  ReadHDF5RealVectorDataset(root, "spline_nodes_phase", & (*submodel)->mf_phi);

  // FIXME: Domain of definition of submodel
  // FIXME: Get these from attributes in the HDF5 file
  // FIXME: check these
  // (*submodel)->q_bounds[0] = 1.0;
  // (*submodel)->q_bounds[1] = 3.0;
  // (*submodel)->chi1_bounds[0] = -0.5;
  // (*submodel)->chi1_bounds[1] = 0.5;
  // (*submodel)->chi2_bounds[0] = -0.5;
  // (*submodel)->chi2_bounds[1] = 0.5;
  // (*submodel)->lambda1_bounds[0] = 0.1;
  // (*submodel)->lambda1_bounds[1] = 5000.0;
  // (*submodel)->lambda2_bounds[0] = 0.1;
  // (*submodel)->lambda2_bounds[1] = 5000.0;

  // FIXME: check consistency of sizes against constants defined above

  // Prepend the point [mf_amp[0], 0] to the phase nodes
  fprintf(stderr, "Before len(mf_phi) = %zu\n", (*submodel)->mf_phi->size);
  double mf_min = gsl_vector_get( (*submodel)->mf_amp, 0); // Follow definition of mf_a in GPSplineSurrogate constructor
  gsl_vector *phi_nodes = gsl_vector_prepend_value((*submodel)->mf_phi, mf_min);
  (*submodel)->mf_phi = phi_nodes;
  fprintf(stderr, "After len(mf_phi) = %zu\n", (*submodel)->mf_phi->size);

  XLALFree(path);
  XLALH5FileClose(file);
  ret = XLAL_SUCCESS;
#else
  XLAL_ERROR(XLAL_EFAILED, "HDF5 support not enabled");
#endif

  return ret;
}

/* Deallocate contents of the given TEOBv4ROMdataDS_submodel structure */
static void TEOBv4ROMdataDS_Cleanup_submodel(TEOBv4ROMdataDS_submodel *submodel) {
  if(submodel->hyp_amp) gsl_matrix_free(submodel->hyp_amp);
  if(submodel->hyp_phi) gsl_matrix_free(submodel->hyp_phi);
  if(submodel->kinv_dot_y_amp) gsl_matrix_free(submodel->kinv_dot_y_amp);
  if(submodel->kinv_dot_y_phi) gsl_matrix_free(submodel->kinv_dot_y_phi);
  if(submodel->x_train) gsl_matrix_free(submodel->x_train);
  if(submodel->mf_amp) gsl_vector_free(submodel->mf_amp);
  if(submodel->mf_phi) gsl_vector_free(submodel->mf_phi);
}

/* Set up a new ROM model, using data contained in dir */
int TEOBv4ROMdataDS_Init(
  UNUSED TEOBv4ROMdataDS *romdata,
  UNUSED const char dir[])
{
  int ret = XLAL_FAILURE;

  /* Create storage for structures */
  if(romdata->setup) {
    XLALPrintError("WARNING: You tried to setup the TEOBv4ROM model that was already initialised. Ignoring\n");
    return (XLAL_FAILURE);
  }

#ifdef LAL_HDF5_ENABLED
  // First, check we got the correct version number
  size_t size = strlen(dir) + strlen(ROMDataHDF5) + 2;
  char *path = XLALMalloc(size);
  snprintf(path, size, "%s/%s", dir, ROMDataHDF5);
  LALH5File *file = XLALH5FileOpen(path, "r");

// FIXME: uncomment this after fixing the attributes in the HDF5 file
  // XLALPrintInfo("ROM metadata\n============\n");
  // PrintInfoStringAttribute(file, "Email");
  // PrintInfoStringAttribute(file, "Description");
  // ret = ROM_check_version_number(file, ROMDataHDF5_VERSION_MAJOR,
  //                                ROMDataHDF5_VERSION_MINOR,
  //                                ROMDataHDF5_VERSION_MICRO);

  XLALFree(path);
  XLALH5FileClose(file);

  ret = TEOBv4ROMdataDS_Init_submodel(&(romdata)->sub1, dir, "sub1");
  if (ret==XLAL_SUCCESS) XLALPrintInfo("%s : submodel 1 loaded successfully.\n", __func__);

  if(XLAL_SUCCESS==ret)
    romdata->setup=1;
  else
    TEOBv4ROMdataDS_Cleanup(romdata);
#else
  XLAL_ERROR(XLAL_EFAILED, "HDF5 support not enabled");
#endif

  return (ret);
}

/* Deallocate contents of the given TEOBv4ROMdataDS structure */
static void TEOBv4ROMdataDS_Cleanup(TEOBv4ROMdataDS *romdata) {
  TEOBv4ROMdataDS_Cleanup_submodel((romdata)->sub1);
  XLALFree((romdata)->sub1);
  (romdata)->sub1 = NULL;
  romdata->setup=0;
}

/* Return the closest higher power of 2  */
// Note: NextPow(2^k) = 2^k for integer values k.
static size_t NextPow2(const size_t n) {
  return 1 << (size_t) ceil(log2(n));
}

static int TaylorF2Phasing(
  double Mtot,         // Total mass in solar masses
  double q,            // Mass-ration m1/m2 >= 1
  double chi1,         // Dimensionless aligned spin of body 1
  double chi2,         // Dimensionless aligned spin of body 2
  double lambda1,      // Tidal deformability of body 1
  double lambda2,      // Tidal deformability of body 2
  double dquadmon1,    // Self-spin deformation of body 1
  double dquadmon2,    // Self-spin deformation of body 2
  gsl_vector *Mfs,     // Input geometric frequencies
  gsl_vector **PNphase // Output: TaylorF2 phase at frequencies Mfs
) {
  XLAL_CHECK(PNphase != NULL, XLAL_EFAULT);
  XLAL_CHECK(*PNphase == NULL, XLAL_EFAULT);
  *PNphase = gsl_vector_alloc(Mfs->size);

  PNPhasingSeries *pn = NULL;
  LALDict *extraParams = XLALCreateDict();
  XLALSimInspiralWaveformParamsInsertPNSpinOrder(extraParams, LAL_SIM_INSPIRAL_SPIN_ORDER_35PN);
  // Ideally, we should be able to specify lambdas like this, but for now the phasing functon does not use the lambda parameters!
  // XLALSimInspiralWaveformParamsInsertTidalLambda1(extraParams, lambda1);
  // XLALSimInspiralWaveformParamsInsertTidalLambda2(extraParams, lambda2);
  // See src/LALSimInspiralTaylorF2.c:XLALSimInspiralTaylorF2AlignedPhasingArray() for how to add the tidal terms
  // We add the terms after the XLALSimInspiralTaylorF2AlignedPhasing() call below to the structure of PN coefficients.

  // FIXME: We should be able to switch on quadrupole-monopole terms (self-spin deformation of the two bodies)
  // from TaylorF2; later these can be replaced with a new TEOB surrogate as well. Just add a switch for now.
  if ((dquadmon1 > 0) || (dquadmon2 > 0)) {
    fprintf(stderr, "Using quadrupole-monopole terms from PN.\n");
    XLALSimInspiralWaveformParamsInsertdQuadMon1(extraParams, dquadmon1);
    XLALSimInspiralWaveformParamsInsertdQuadMon2(extraParams, dquadmon2);
  }

  double m1OverM = q / (1.0+q);
  double m2OverM = 1.0 / (1.0+q);
  double m1 = Mtot * m1OverM * LAL_MSUN_SI;
  double m2 = Mtot * m2OverM * LAL_MSUN_SI;
  XLALSimInspiralTaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2, extraParams);

// fprintf(stderr, "%g %g %g %g %g\n", Mtot, m1, m2, chi1, chi2);

  // Add tidal deformability terms
  pn->v[10] = pn->v[0] * ( lambda1 * XLALSimInspiralTaylorF2Phasing_10PNTidalCoeff(m1OverM)
                         + lambda2 * XLALSimInspiralTaylorF2Phasing_10PNTidalCoeff(m2OverM) );
  pn->v[12] = pn->v[0] * ( lambda1 * XLALSimInspiralTaylorF2Phasing_12PNTidalCoeff(m1OverM)
                         + lambda2 * XLALSimInspiralTaylorF2Phasing_12PNTidalCoeff(m2OverM) );

// fprintf(stderr, "pn->v: %g %g %g %g %g %g %g %g \n", pn->v[7], pn->v[6], pn->v[5], pn->v[4], pn->v[3], pn->v[2], pn->v[1], pn->v[0]);
// fprintf(stderr, "pn->vlogv: %g %g\n", pn->vlogv[6], pn->vlogv[5]);

// fprintf(stderr, "\nPN phasing at nodes:");
  for (size_t i=0; i < Mfs->size; i++) {
      const double Mf = gsl_vector_get(Mfs, i);
      const double v = cbrt(LAL_PI * Mf);
      const double logv = log(v);
      // FIXME: optimize this further: v4=v2*v2, v8=v4*v4
      const double v2 = v * v;
      const double v3 = v * v2;
      const double v4 = v * v3;
      const double v5 = v * v4;
      const double v6 = v * v5;
      const double v7 = v * v6;
      const double v8 = v * v7;
      const double v9 = v * v8;
      const double v10 = v * v9;
      const double v12 = v2 * v10;
      double phasing = 0.0;

      phasing += pn->v[7] * v7;
      phasing += (pn->v[6] + pn->vlogv[6] * logv) * v6;
      phasing += (pn->v[5] + pn->vlogv[5] * logv) * v5;
      phasing += pn->v[4] * v4;
      phasing += pn->v[3] * v3;
      phasing += pn->v[2] * v2;
      phasing += pn->v[1] * v;
      phasing += pn->v[0];

      /* Tidal terms in phasing */
      phasing += pn->v[12] * v12;
      phasing += pn->v[10] * v10;

      phasing /= v5;
      // LALSimInspiralTaylorF2.c
      // shft = LAL_TWOPI * (tC.gpsSeconds + 1e-9 * tC.gpsNanoSeconds); // FIXME: this should be done in the main generator and may already be there
      // phasing += shft * f - 2.*phi_ref - ref_phasing; // FIXME: add ref

      gsl_vector_set(*PNphase, i, -phasing);
      // fprintf(stderr, "%.15g\n", phasing);

      //fprintf(stderr, "PN phasing[%zu] = %g, %g\n", i, Mf, phasing);
      //gsl_vector_set(phi_at_nodes, i, phasing + gsl_vector_get(sur_phi_at_nodes, i));
      // TODO: compare total and all the terms
  }
  // fprintf(stderr, "\n");
  //amp * cos(phasing - LAL_PI_4) - amp * sin(phasing - LAL_PI_4) * 1.0j;
 // amp0 = -4. * m1 * m2 / r * LAL_MRSUN_SI * LAL_MTSUN_SI * sqrt(LAL_PI/12.L);

  XLALDestroyDict(extraParams);
  XLALFree(pn);

  return XLAL_SUCCESS;
}

// 1PN point-particle amplitude.
// Expression from Eq. (6.10) of arXiv:0810.5336.
// !!! This is technically wrong since you have a x**2 term (need to re-expand). !!!
static int TaylorF2Amplitude1PN(
  double eta,        // Symmetric mass-ratio
  gsl_vector *Mfs,   // Input geometric frequencies
  gsl_vector **PNamp // Output: TaylorF2 amplitude at frequencies Mfs
) {
  XLAL_CHECK(PNamp != NULL, XLAL_EFAULT);
  XLAL_CHECK(*PNamp == NULL, XLAL_EFAULT);
  *PNamp = gsl_vector_alloc(Mfs->size);

  for (size_t i=0; i < Mfs->size; i++) {
    const double Mf = gsl_vector_get(Mfs, i);
    const double v = cbrt(LAL_PI * Mf);
    const double x = v*v;

    double a00 = sqrt( (5.0*LAL_PI/24.0)*eta );
    double a10 = -323.0/224.0 + 451.0*eta/168.0;
    double amp =  2.0*a00 * pow(x, -7.0/4.0) * (1.0 + a10*x);
    gsl_vector_set(*PNamp, i, amp);
  }

  return XLAL_SUCCESS;
}

/**
 * Core function for computing the ROM waveform.
 * Interpolate projection coefficient data and evaluate coefficients at desired (q, chi1, chi2).
 * Construct 1D splines for amplitude and phase.
 * Compute strain waveform from amplitude and phase.
*/
static int TEOBv4ROMCore(
  COMPLEX16FrequencySeries **hptilde,
  COMPLEX16FrequencySeries **hctilde,
  double phiRef, // orbital reference phase
  double fRef,
  double distance,
  double inclination,
  double Mtot_sec,
  double eta,
  double chi1,
  double chi2,
  double lambda1,
  double lambda2,
  const REAL8Sequence *freqs_in, /* Frequency points at which to evaluate the waveform (Hz) */
  double deltaF
  /* If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  )
{
  /* Check output arrays */
  if(!hptilde || !hctilde)
    XLAL_ERROR(XLAL_EFAULT);

  TEOBv4ROMdataDS *romdata=&__lalsim_TEOBv4ROMDS_data;
  if (!TEOBv4ROM_IsSetup()) {
    XLAL_ERROR(XLAL_EFAILED,
               "Error setting up TEOBv4ROM data - check your $LAL_DATA_PATH\n");
  }

  if(*hptilde || *hctilde) {
    XLALPrintError("(*hptilde) and (*hctilde) are supposed to be NULL, but got %p and %p",
                   (*hptilde), (*hctilde));
    XLAL_ERROR(XLAL_EFAULT);
  }
  int retcode=0;

  double Mtot = Mtot_sec / LAL_MTSUN_SI;

  // 'Nudge' parameter values to allowed boundary values if close by
  if (eta > 0.25)     nudge(&eta, 0.25, 1e-6);
  if (eta < 0.01)     nudge(&eta, 0.01, 1e-6);

  if (chi1 < -1.0 || chi2 < -1.0 || chi1 > 1.0 || chi2 > 1.0) {
    XLALPrintError("XLAL Error - %s: chi1 or chi2 smaller than -1.0 or larger than 1.0!\n"
                   "TEOBv4ROM is only available for spins in the range -1 <= a/M <= 1.0.\n",
                   __func__);
    XLAL_ERROR( XLAL_EDOM );
  }

  if (eta<0.01 || eta > 0.25) {
    XLALPrintError("XLAL Error - %s: eta (%f) smaller than 0.01 or unphysical!\n"
                   "TEOBv4ROM is only available for eta in the range 0.01 <= eta <= 0.25.\n",
                   __func__, eta);
    XLAL_ERROR( XLAL_EDOM );
  }

  TEOBv4ROMdataDS_submodel *submodel_lo; // FIXME: rename submodel_lo to model
  submodel_lo = romdata->sub1;

  /* Find frequency bounds */
  if (!freqs_in) XLAL_ERROR(XLAL_EFAULT);
  double fLow  = freqs_in->data[0];
  double fHigh = freqs_in->data[freqs_in->length - 1];

  if(fRef==0.0)
    fRef=fLow;

  /* Convert to geometric units for frequency */
  int N_amp = submodel_lo->mf_amp->size;
  int N_phi = submodel_lo->mf_phi->size;

  // lowest allowed geometric frequency for ROM
  double Mf_ROM_min = fmax(gsl_vector_get(submodel_lo->mf_amp, 0),
                           gsl_vector_get(submodel_lo->mf_phi,0));
  // highest allowed geometric frequency for ROM
  double Mf_ROM_max = fmin(gsl_vector_get(submodel_lo->mf_amp, N_amp-1),
                           gsl_vector_get(submodel_lo->mf_phi, N_phi-1));
  double fLow_geom = fLow * Mtot_sec;
  double fHigh_geom = fHigh * Mtot_sec;
  double fRef_geom = fRef * Mtot_sec;
  double deltaF_geom = deltaF * Mtot_sec;

  // Enforce allowed geometric frequency range
  if (fLow_geom < Mf_ROM_min)
    XLAL_ERROR(XLAL_EDOM, "Starting frequency Mflow=%g is smaller than lowest frequency in ROM Mf=%g.\n", fLow_geom, Mf_ROM_min);
  if (fHigh_geom == 0 || fHigh_geom > Mf_ROM_max)
    fHigh_geom = Mf_ROM_max;
  else if (fHigh_geom < Mf_ROM_min)
    XLAL_ERROR(XLAL_EDOM, "End frequency %g is smaller than ROM starting frequency %g!\n", fHigh_geom, Mf_ROM_min);
  if (fHigh_geom <= fLow_geom)
    XLAL_ERROR(XLAL_EDOM, "End frequency %g is smaller than (or equal to) starting frequency %g!\n", fHigh_geom, fLow_geom);
  if (fRef_geom > Mf_ROM_max) {
    XLALPrintWarning("Reference frequency Mf_ref=%g is greater than maximal frequency in ROM Mf=%g. Starting at maximal frequency in ROM.\n", fRef_geom, Mf_ROM_max);
    fRef_geom = Mf_ROM_max; // If fref > fhigh we reset fref to default value of cutoff frequency.
  }
  if (fRef_geom < Mf_ROM_min) {
    XLALPrintWarning("Reference frequency Mf_ref=%g is smaller than lowest frequency in ROM Mf=%g. Starting at lowest frequency in ROM.\n", fLow_geom, Mf_ROM_min);
    fRef_geom = Mf_ROM_min;
  }

  if (Mtot_sec/LAL_MTSUN_SI > 500.0)
    XLALPrintWarning("Total mass=%gMsun > 500Msun. TEOBv4ROM disagrees with SEOBNRv4 for high total masses.\n", Mtot_sec/LAL_MTSUN_SI);


  // Evaluate GPR for log amplitude and dephasing
  double q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  gsl_vector *sur_amp_at_nodes = gsl_vector_alloc(N_amp);
  gsl_vector *sur_phi_at_nodes_tmp = gsl_vector_alloc(N_phi - 1); // Will prepend a point below
  assert(N_amp == N_phi); // FIXME: is it safe to assume this?

  retcode = GPR_evaluation_5D(
    q, chi1, chi2, lambda1, lambda2,
    submodel_lo->hyp_amp,
    submodel_lo->hyp_phi,
    submodel_lo->kinv_dot_y_amp,
    submodel_lo->kinv_dot_y_phi,
    submodel_lo->x_train,
    sur_amp_at_nodes,
    sur_phi_at_nodes_tmp
  );

  if(retcode!=0) {
    //TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_lo); /// FIXME: change to clean the data for the GPR model
    XLAL_ERROR(retcode);
  }

  for (int i=0; i<N_phi-1; i++)
    fprintf(stderr, "mf_phi, phi_at_nodes [%d] = (%g, %g)\n", i,
    gsl_vector_get(submodel_lo->mf_phi, i),
    gsl_vector_get(sur_phi_at_nodes_tmp, i));

  // Prepend the point [mf_min, 0] to the phase nodes
  // This has already been done in the setup for mf_phi
  gsl_vector *sur_phi_at_nodes = gsl_vector_prepend_value(sur_phi_at_nodes_tmp, 0.0);
fprintf(stderr, "phi_at_nodes->size = %zu\n", sur_phi_at_nodes->size);
fprintf(stderr, "N_amp, N_phi = %d, %d\n", N_amp, N_phi);
  for (int i=0; i<N_phi; i++)
    fprintf(stderr, "mf_phi, phi_at_nodes [%d] = (%g, %g)\n", i,
    gsl_vector_get(submodel_lo->mf_phi, i),
    gsl_vector_get(sur_phi_at_nodes, i));


double dquadmon1 = 0.0; // FIXME
double dquadmon2 = 0.0;
gsl_vector *PN_phi_at_nodes = NULL;
TaylorF2Phasing(Mtot, q, chi1, chi2, lambda1, lambda2, dquadmon1, dquadmon2, submodel_lo->mf_phi, &PN_phi_at_nodes);

fprintf(stderr, "\nphiPN_at_nodes:");
gsl_vector_fprintf(stderr, PN_phi_at_nodes, "%.15g");


// FIXME: copy submodel_lo->mf_phi to a dedicated vector
gsl_vector *PN_amp_at_nodes = NULL;
TaylorF2Amplitude1PN(eta, submodel_lo->mf_phi, &PN_amp_at_nodes); // FIXME: should input mf_amp unless it is the same as mf_phi

fprintf(stderr, "\nampPN_at_nodes:");
gsl_vector_fprintf(stderr, PN_amp_at_nodes, "%.15g");



// Setup 1d splines in frequency
gsl_interp_accel *acc_phi = gsl_interp_accel_alloc();
gsl_spline *spline_phi = gsl_spline_alloc(gsl_interp_cspline, N_phi);
gsl_vector_add(sur_phi_at_nodes, PN_phi_at_nodes); // stores result in sur_phi_at_nodes
gsl_spline_init(spline_phi, gsl_vector_const_ptr(submodel_lo->mf_phi, 0),
                gsl_vector_const_ptr(sur_phi_at_nodes, 0), N_phi);



gsl_interp_accel *acc_amp = gsl_interp_accel_alloc();
gsl_spline *spline_amp = gsl_spline_alloc(gsl_interp_cspline, N_amp);
// Compute amplitude = PN_amplitude * exp(surrogate_amplitude)
gsl_vector *spline_amp_values = gsl_vector_alloc(N_amp);
for (int i=0; i<N_amp; i++) {
  double amp_i = gsl_vector_get(PN_amp_at_nodes, i) * exp(gsl_vector_get(sur_amp_at_nodes, i));
  gsl_vector_set(spline_amp_values, i, amp_i);
}

gsl_spline_init(spline_amp, gsl_vector_const_ptr(submodel_lo->mf_phi, 0), // FIXME: should input mf_amp unless it is the same as mf_phi
                gsl_vector_const_ptr(spline_amp_values, 0), N_amp);





  size_t npts = 0;
  LIGOTimeGPS tC = {0, 0};
  UINT4 offset = 0; // Index shift between freqs and the frequency series
  REAL8Sequence *freqs = NULL;
  if (deltaF > 0)  { // freqs contains uniform frequency grid with spacing deltaF; we start at frequency 0
    /* Set up output array with size closest power of 2 */
    npts = NextPow2(fHigh_geom / deltaF_geom) + 1;
    if (fHigh_geom < fHigh * Mtot_sec) /* Resize waveform if user wants f_max larger than cutoff frequency */
      npts = NextPow2(fHigh * Mtot_sec / deltaF_geom) + 1;

    XLALGPSAdd(&tC, -1. / deltaF);  /* coalesce at t=0 */
    *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &tC, 0.0, deltaF, &lalStrainUnit, npts);
    *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &tC, 0.0, deltaF, &lalStrainUnit, npts);

    // Recreate freqs using only the lower and upper bounds
    // Use fLow, fHigh and deltaF rather than geometric frequencies for numerical accuracy
    double fHigh_temp = fHigh_geom / Mtot_sec;
    UINT4 iStart = (UINT4) ceil(fLow / deltaF);
    UINT4 iStop = (UINT4) ceil(fHigh_temp / deltaF);
    freqs = XLALCreateREAL8Sequence(iStop - iStart);
    if (!freqs) {
      XLAL_ERROR(XLAL_EFUNC, "Frequency array allocation failed.");
    }
    for (UINT4 i=iStart; i<iStop; i++)
      freqs->data[i-iStart] = i*deltaF_geom;

    offset = iStart;
  } else { // freqs contains frequencies with non-uniform spacing; we start at lowest given frequency
    npts = freqs_in->length;
    *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &tC, fLow, 0, &lalStrainUnit, npts);
    *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &tC, fLow, 0, &lalStrainUnit, npts);
    offset = 0;

    freqs = XLALCreateREAL8Sequence(freqs_in->length);
    if (!freqs) {
      XLAL_ERROR(XLAL_EFUNC, "Frequency array allocation failed.");
    }
    for (UINT4 i=0; i<freqs_in->length; i++)
      freqs->data[i] = freqs_in->data[i] * Mtot_sec;
  }

  if (!(*hptilde) || !(*hctilde))	{
      XLALDestroyREAL8Sequence(freqs);
      gsl_spline_free(spline_amp);
      gsl_spline_free(spline_phi);
      gsl_interp_accel_free(acc_amp);
      gsl_interp_accel_free(acc_phi);
      XLAL_ERROR(XLAL_EFUNC);
  }
  memset((*hptilde)->data->data, 0, npts * sizeof(COMPLEX16));
  memset((*hctilde)->data->data, 0, npts * sizeof(COMPLEX16));

  XLALUnitMultiply(&(*hptilde)->sampleUnits, &(*hptilde)->sampleUnits, &lalSecondUnit);
  XLALUnitMultiply(&(*hctilde)->sampleUnits, &(*hctilde)->sampleUnits, &lalSecondUnit);

  COMPLEX16 *pdata=(*hptilde)->data->data;
  COMPLEX16 *cdata=(*hctilde)->data->data;

  REAL8 cosi = cos(inclination);
  REAL8 pcoef = 0.5*(1.0 + cosi*cosi);
  REAL8 ccoef = cosi;

  REAL8 s = 0.5; // Scale polarization amplitude so that strain agrees with FFT of SEOBNRv4
  double amp0 = Mtot * Mtot_sec * LAL_MRSUN_SI / (distance); // Correct overall amplitude to undo mass-dependent scaling used in ROM

  // Evaluate reference phase for setting phiRef correctly
  double phase_change = gsl_spline_eval(spline_phi, fRef_geom, acc_phi) - 2*phiRef;

  // Compute BNS merger frequency
  TidalEOBParams tidal1, tidal2;
  tidal1.mByM = q / (1.0+q);
  tidal1.lambda2Tidal = lambda1 * pow(tidal1.mByM,5);
  tidal2.mByM = 1.0 / (1.0+q);
  tidal2.lambda2Tidal = lambda2 * pow(tidal2.mByM,5);
  double Momega22_BNS_mrg = XLALSimNSNSMergerFreq(&tidal1, &tidal2);
  fprintf(stderr, "Momega22_BNS_mrg = %g\n", Momega22_BNS_mrg);

  Mf_ROM_max = gsl_vector_get(submodel_lo->mf_phi, N_phi-1);
  double Mf_final = fmin(Momega22_BNS_mrg, Mf_ROM_max);

  // Assemble waveform from aplitude and phase
  fprintf(stderr, "Mf_ROM_max = %g\n", Mf_ROM_max);
  for (UINT4 i=0; i<freqs->length; i++) { // loop over frequency points in sequence
    double f = freqs->data[i];
    if (f > Mf_final) continue; // We're beyond the highest allowed frequency; since freqs may not be ordered, we'll just skip the current frequency and leave zero in the buffer
    int j = i + offset; // shift index for frequency series if needed
    double A = gsl_spline_eval(spline_amp, f, acc_amp);
    double phase = gsl_spline_eval(spline_phi, f, acc_phi) - phase_change;
    fprintf(stderr, "%d, %d    %g : %g %g\n", i, j, f, A, phase);
    COMPLEX16 htilde = s*amp0*A * (cos(phase) + I*sin(phase));//cexp(I*phase);
    pdata[j] =      pcoef * htilde;
    cdata[j] = -I * ccoef * htilde;
  }

  /* Correct phasing so we coalesce at t=0 (with the definition of the epoch=-1/deltaF above) */
  UINT4 L = freqs->length;
  // prevent gsl interpolation errors
  if (Mf_final > freqs->data[L-1])
    Mf_final = freqs->data[L-1];
  if (Mf_final < freqs->data[0]) {
    XLALDestroyREAL8Sequence(freqs);
    gsl_spline_free(spline_amp);
    gsl_spline_free(spline_phi);
    gsl_interp_accel_free(acc_amp);
    gsl_interp_accel_free(acc_phi);
    XLAL_ERROR(XLAL_EDOM, "f_ringdown < f_min");
  }

  // Time correction is t(f_final) = 1/(2pi) dphi/df (f_final)
  // We compute the dimensionless time correction t/M since we use geometric units.
  REAL8 t_corr = gsl_spline_eval_deriv(spline_phi, Mf_final, acc_phi) / (2*LAL_PI);

  // Now correct phase
  for (UINT4 i=0; i<freqs->length; i++) { // loop over frequency points in sequence
    double f = freqs->data[i] - fRef_geom;
    int j = i + offset; // shift index for frequency series if needed
    double phase_factor = -2*LAL_PI * f * t_corr;
    COMPLEX16 t_factor = (cos(phase_factor) + I*sin(phase_factor));//cexp(I*phase_factor);
    pdata[j] *= t_factor;
    cdata[j] *= t_factor;
  }

  XLALDestroyREAL8Sequence(freqs);

  gsl_interp_accel_free(acc_phi);
  gsl_spline_free(spline_phi);
  gsl_interp_accel_free(acc_amp);
  gsl_spline_free(spline_amp);

  return(XLAL_SUCCESS);
}

/**
 * @addtogroup LALSimIMRTEOBv4ROM_c
 *
 * \author Michael Puerrer, Ben Lackey
 *
 * \brief C code for TEOBv4 reduced order model
 * See arXiv:xxxxxxxxxxxx
 *
 * This is a frequency domain model that approximates the time domain TEOBv4 model.
 *
 * The binary data HDF5 file (TEOBv4ROM_v1.0.hdf5)
 * will be available at on LIGO clusters in /home/cbc/.
 * Make sure the files are in your LAL_DATA_PATH.
 *
 * @note Note that due to its construction the iFFT of the ROM has a small (~ 20 M) offset
 * in the peak time that scales with total mass as compared to the time-domain TEOBv4 model.
 *
 * @note Parameter ranges: FIXME
 *   * q
 *   * chi_i
 *   * lambda_i
 *   * 2Msun (@ flow=20Hz) <= Mtot
 *
 *  Aligned component spins chi1, chi2.
 *  Symmetric mass-ratio eta = m1*m2/(m1+m2)^2.
 *  Total mass Mtot.
 *
 * @{
 */


/**
 * Compute waveform in LAL format at specified frequencies for the TEOBv4_ROM model.
 *
 * XLALSimIMRTEOBv4ROM() returns the plus and cross polarizations as a complex
 * frequency series with equal spacing deltaF and contains zeros from zero frequency
 * to the starting frequency and zeros beyond the cutoff frequency in the ringdown.
 *
 * In contrast, XLALSimIMRTEOBv4ROMFrequencySequence() returns a
 * complex frequency series with entries exactly at the frequencies specified in
 * the sequence freqs (which can be unequally spaced). No zeros are added.
 *
 * If XLALSimIMRTEOBv4ROMFrequencySequence() is called with frequencies that
 * are beyond the maxium allowed geometric frequency for the ROM, zero strain is returned.
 * It is not assumed that the frequency sequence is ordered.
 *
 * This function is designed as an entry point for reduced order quadratures.
 */
int XLALSimIMRTEOBv4ROMFrequencySequence(
  struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
  struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
  const REAL8Sequence *freqs,                   /**< Frequency points at which to evaluate the waveform (Hz) */
  REAL8 phiRef,                                 /**< Orbital phase at reference time */
  REAL8 fRef,                                   /**< Reference frequency (Hz); 0 defaults to fLow */
  REAL8 distance,                               /**< Distance of source (m) */
  REAL8 inclination,                            /**< Inclination of source (rad) */
  REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
  REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
  REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
  REAL8 chi2,                                   /**< Dimensionless aligned component spin 2 */
  REAL8 lambda1,                                /**< Dimensionless tidal deformability 1 */
  REAL8 lambda2)                                /**< Dimensionless tidal deformability 2 */
{
  /* Internally we need m1 > m2, so change around if this is not the case */
  if (m1SI < m2SI) {
    // Swap m1 and m2
    double m1temp = m1SI;
    double chi1temp = chi1;
    m1SI = m2SI;
    chi1 = chi2;
    m2SI = m1temp;
    chi2 = chi1temp;
  }

  /* Get masses in terms of solar mass */
  double mass1 = m1SI / LAL_MSUN_SI;
  double mass2 = m2SI / LAL_MSUN_SI;
  double Mtot = mass1+mass2;
  double eta = mass1 * mass2 / (Mtot*Mtot);  /* Symmetric mass-ratio */
  double Mtot_sec = Mtot * LAL_MTSUN_SI;     /* Total mass in seconds */

  if (!freqs) XLAL_ERROR(XLAL_EFAULT);

  // Load ROM data if not loaded already
#ifdef LAL_PTHREAD_LOCK
  (void) pthread_once(&TEOBv4ROM_is_initialized, TEOBv4ROM_Init_LALDATA);
#else
  TEOBv4ROM_Init_LALDATA();
#endif

  if(!TEOBv4ROM_IsSetup()) {
    XLAL_ERROR(XLAL_EFAILED,
               "Error setting up TEOBv4ROM data - check your $LAL_DATA_PATH\n");
  }

  // Call the internal core function with deltaF = 0 to indicate that freqs is non-uniformly
  // spaced and we want the strain only at these frequencies
  int retcode = TEOBv4ROMCore(hptilde, hctilde, phiRef, fRef, distance,
                                inclination, Mtot_sec, eta, chi1, chi2, lambda1, lambda2, freqs, 0);

  return(retcode);
}

/**
 * Compute waveform in LAL format for the SEOBNRv4_ROM model.
 *
 * Returns the plus and cross polarizations as a complex frequency series with
 * equal spacing deltaF and contains zeros from zero frequency to the starting
 * frequency fLow and zeros beyond the cutoff frequency in the ringdown.
 */
int XLALSimIMRTEOBv4ROM(
  struct tagCOMPLEX16FrequencySeries **hptilde, /**< Output: Frequency-domain waveform h+ */
  struct tagCOMPLEX16FrequencySeries **hctilde, /**< Output: Frequency-domain waveform hx */
  REAL8 phiRef,                                 /**< Phase at reference time */
  REAL8 deltaF,                                 /**< Sampling frequency (Hz) */
  REAL8 fLow,                                   /**< Starting GW frequency (Hz) */
  REAL8 fHigh,                                  /**< End frequency; 0 defaults to Mf=0.14 */
  REAL8 fRef,                                   /**< Reference frequency (Hz); 0 defaults to fLow */
  REAL8 distance,                               /**< Distance of source (m) */
  REAL8 inclination,                            /**< Inclination of source (rad) */
  REAL8 m1SI,                                   /**< Mass of companion 1 (kg) */
  REAL8 m2SI,                                   /**< Mass of companion 2 (kg) */
  REAL8 chi1,                                   /**< Dimensionless aligned component spin 1 */
  REAL8 chi2,                                   /**< Dimensionless aligned component spin 2 */
  REAL8 lambda1,                                /**< Dimensionless tidal deformability 1 */
  REAL8 lambda2)                                /**< Dimensionless tidal deformability 2 */
{
  /* Internally we need m1 > m2, so change around if this is not the case */
  if (m1SI < m2SI) {
    // Swap m1 and m2
    double m1temp = m1SI;
    double chi1temp = chi1;
    m1SI = m2SI;
    chi1 = chi2;
    m2SI = m1temp;
    chi2 = chi1temp;
  }

  /* Get masses in terms of solar mass */
  double mass1 = m1SI / LAL_MSUN_SI;
  double mass2 = m2SI / LAL_MSUN_SI;
  double Mtot = mass1+mass2;
  double eta = mass1 * mass2 / (Mtot*Mtot);    /* Symmetric mass-ratio */
  double Mtot_sec = Mtot * LAL_MTSUN_SI;       /* Total mass in seconds */

  if(fRef==0.0)
    fRef=fLow;

  // Load ROM data if not loaded already
#ifdef LAL_PTHREAD_LOCK
  (void) pthread_once(&TEOBv4ROM_is_initialized, TEOBv4ROM_Init_LALDATA);
#else
  TEOBv4ROM_Init_LALDATA();
#endif

  // Use fLow, fHigh, deltaF to compute freqs sequence
  // Instead of building a full sequency we only transfer the boundaries and let
  // the internal core function do the rest (and properly take care of corner cases).
  REAL8Sequence *freqs = XLALCreateREAL8Sequence(2);
  freqs->data[0] = fLow;
  freqs->data[1] = fHigh;

  int retcode = TEOBv4ROMCore(hptilde, hctilde, phiRef, fRef, distance,
                                inclination, Mtot_sec, eta, chi1, chi2, lambda1, lambda2, freqs, deltaF);

  XLALDestroyREAL8Sequence(freqs);

  return(retcode);
}

/** @} */


/** Setup TEOBv4ROM model using data files installed in $LAL_DATA_PATH
 */
UNUSED static void TEOBv4ROM_Init_LALDATA(void)
{
  fprintf(stderr, "In TEOBv4ROM_Init_LALDATA()\n");
  if (TEOBv4ROM_IsSetup()) return;

  // Expect ROM datafile in a directory listed in LAL_DATA_PATH,
#ifdef LAL_HDF5_ENABLED
#define datafile ROMDataHDF5
  char *path = XLALFileResolvePathLong(datafile, PKG_DATA_DIR);
  if (path==NULL)
    XLAL_ERROR_VOID(XLAL_EIO, "Unable to resolve data file %s in $LAL_DATA_PATH\n", datafile);
  char *dir = dirname(path);
  int ret = TEOBv4ROM_Init(dir);
  XLALFree(path);

  if(ret!=XLAL_SUCCESS)
    XLAL_ERROR_VOID(XLAL_FAILURE, "Unable to find TEOBv4ROM data files in $LAL_DATA_PATH\n");
#else
  XLAL_ERROR_VOID(XLAL_EFAILED, "TEOBv4ROM requires HDF5 support which is not enabled\n");
#endif
}
