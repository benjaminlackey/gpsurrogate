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
static const char ROMDataHDF5[] = "SEOBNRv4ROM_v2.0.hdf5";
static const INT4 ROMDataHDF5_VERSION_MAJOR = 2;
static const INT4 ROMDataHDF5_VERSION_MINOR = 0;
static const INT4 ROMDataHDF5_VERSION_MICRO = 0;
#endif

#include <lal/LALSimInspiral.h>
#include <lal/LALSimIMR.h>

#include "LALSimInspiralPNCoefficients.c"
#include "LALSimIMRSEOBNRROMUtilities.c"

#include <lal/LALConfig.h>
#ifdef LAL_PTHREAD_LOCK
#include <pthread.h>
#endif


// TODO:
// D add lambda1,2 args
// D add prototypes
// * remove the glueing functions -- will be easier to copy logic from earlier code
// D test reading of HDF5 test data
// D implement code to compute kernel and gpr
// * compute and check amp,phi at EIM nodes
// * assemble waveform and check against python output
// * add waveform approximant and glueing code in LALSimInspiral
// remove all debugging code
// add more checks and tests


#ifdef LAL_PTHREAD_LOCK
static pthread_once_t TEOBv4ROM_is_initialized = PTHREAD_ONCE_INIT;
#endif

/*************** type definitions ******************/

typedef struct tagTEOBv4ROMdataDS_coeff
{
  gsl_vector* c_amp;
  gsl_vector* c_phi;
} TEOBv4ROMdataDS_coeff;

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
  // gsl_matrix *B_amp;           // Reduced basis for log amplitude
  // gsl_matrix *B_phi;           // Reduced basis for dephasing
  gsl_matrix *x_train;         // Training points
  gsl_vector *mf_amp;          // location of spline nodes for log amplitude
  gsl_vector *mf_phi;          // location of spline nodes for dephasing

  // 5D parameter space bounds of surrogate
  double q_bounds[2];          // [q_min, q_max]
  double chi1_bounds[2];       // [chi1_min, chi1_max]
  double chi2_bounds[2];       // [chi2_min, chi2_max]
  double lambda1_bounds[2];    // [lambda1_min, lambda1_max]
  double lambda2_bounds[2];    // [lambda2_min, lambda2_max]

  gsl_vector* cvec_amp;      // Flattened amplitude projection coefficients
  gsl_vector* cvec_phi;      // Flattened phase projection coefficients
  gsl_matrix *Bamp;          // Reduced SVD basis for amplitude
  gsl_matrix *Bphi;          // Reduced SVD basis for phase
  int nk_amp;                // Number frequency points for amplitude
  int nk_phi;                // Number of frequency points for phase
  gsl_vector *gA;            // Sparse frequency points for amplitude
  gsl_vector *gPhi;          // Sparse frequency points for phase
  gsl_vector *etavec;        // B-spline knots in eta
  gsl_vector *chi1vec;       // B-spline knots in chi1
  gsl_vector *chi2vec;       // B-spline knots in chi2
  int ncx, ncy, ncz;         // Number of points in eta, chi1, chi2
  double eta_bounds[2];      // [eta_min, eta_max]
};
typedef struct tagTEOBv4ROMdataDS_submodel TEOBv4ROMdataDS_submodel;

struct tagTEOBv4ROMdataDS
{
  UINT4 setup;
  TEOBv4ROMdataDS_submodel* sub1;
  TEOBv4ROMdataDS_submodel* sub2;
  TEOBv4ROMdataDS_submodel* sub3;
};
typedef struct tagTEOBv4ROMdataDS TEOBv4ROMdataDS;

static TEOBv4ROMdataDS __lalsim_TEOBv4ROMDS_data;

typedef int (*load_dataPtr)(const char*, gsl_vector *, gsl_vector *, gsl_matrix *, gsl_matrix *, gsl_vector *);

typedef struct tagSplineData
{
  gsl_bspline_workspace *bwx;
  gsl_bspline_workspace *bwy;
  gsl_bspline_workspace *bwz;
} SplineData;

/**************** Internal functions **********************/

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

static int TP_Spline_interpolation_3d(
  REAL8 eta,                // Input: eta-value for which projection coefficients should be evaluated
  REAL8 chi1,               // Input: chi1-value for which projection coefficients should be evaluated
  REAL8 chi2,               // Input: chi2-value for which projection coefficients should be evaluated
  gsl_vector *cvec_amp,     // Input: data for spline coefficients for amplitude
  gsl_vector *cvec_phi,     // Input: data for spline coefficients for phase
//  gsl_vector *cvec_amp_pre, // Input: data for spline coefficients for amplitude prefactor
  int nk_amp,               // number of SVD-modes == number of basis functions for amplitude
  int nk_phi,               // number of SVD-modes == number of basis functions for phase
  int nk_max,               // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
  int ncx,                  // Number of points in eta  + 2
  int ncy,                  // Number of points in chi1 + 2
  int ncz,                  // Number of points in chi2 + 2
  const double *etavec,     // B-spline knots in eta
  const double *chi1vec,    // B-spline knots in chi1
  const double *chi2vec,    // B-spline knots in chi2
  gsl_vector *c_amp,        // Output: interpolated projection coefficients for amplitude
  gsl_vector *c_phi         // Output: interpolated projection coefficients for phase
//  REAL8 *amp_pre            // Output: interpolated amplitude prefactor
);

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
  double deltaF,
  /* If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  int nk_max // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
);

UNUSED static void TEOBv4ROMdataDS_coeff_Init(TEOBv4ROMdataDS_coeff **romdatacoeff, int nk_amp, int nk_phi);
UNUSED static void TEOBv4ROMdataDS_coeff_Cleanup(TEOBv4ROMdataDS_coeff *romdatacoeff);

static size_t NextPow2(const size_t n);
UNUSED static void SplineData_Destroy(SplineData *splinedata);
UNUSED static void SplineData_Init(
  SplineData **splinedata,
  int ncx,                // Number of points in eta  + 2
  int ncy,                // Number of points in chi1 + 2
  int ncz,                // Number of points in chi2 + 2
  const double *etavec,   // B-spline knots in eta
  const double *chi1vec,  // B-spline knots in chi1
  const double *chi2vec   // B-spline knots in chi2
);

UNUSED static int TEOBv4ROMTimeFrequencySetup(
  gsl_spline **spline_phi,                      // phase spline
  gsl_interp_accel **acc_phi,                   // phase spline accelerator
  REAL8 *Mf_final,                              // ringdown frequency in Mf
  REAL8 *Mtot_sec,                              // total mass in seconds
  REAL8 m1SI,                                   // Mass of companion 1 (kg)
  REAL8 m2SI,                                   // Mass of companion 2 (kg)
  REAL8 chi1,                                   // Aligned spin of companion 1
  REAL8 chi2,                                   // Aligned spin of companion 2
  REAL8 *Mf_ROM_min,                            // Lowest geometric frequency for ROM
  REAL8 *Mf_ROM_max                             // Highest geometric frequency for ROM
);

UNUSED static REAL8 Interpolate_Coefficent_Matrix(
  gsl_vector *v,
  REAL8 eta,
  REAL8 chi,
  int ncx,
  int ncy,
  gsl_bspline_workspace *bwx,
  gsl_bspline_workspace *bwy
);

UNUSED static void GlueAmplitude(
  // INPUTS
  TEOBv4ROMdataDS_submodel *submodel_lo,
  TEOBv4ROMdataDS_submodel *submodel_hi,
  gsl_vector* amp_f_lo,
  gsl_vector* amp_f_hi,
  double amp_pre_lo,
  double amp_pre_hi,
  const double Mfm,
  // OUTPUTS
  gsl_interp_accel **acc_amp,
  gsl_spline **spline_amp
);

UNUSED static void GluePhasing(
  // INPUTS
  TEOBv4ROMdataDS_submodel *submodel_lo,
  TEOBv4ROMdataDS_submodel *submodel_hi,
  gsl_vector* phi_f_lo,
  gsl_vector* phi_f_hi,
  const double Mfm,
  // OUTPUTS
  gsl_interp_accel **acc_phi_out,
  gsl_spline **spline_phi_out
);

static gsl_vector *gsl_vector_prepend_value(gsl_vector *v, double value);

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
fprintf(stderr, "In TEOBv4ROM_Init()\n");
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

// Setup B-spline basis functions for given points
static void SplineData_Init(
  SplineData **splinedata,
  int ncx,                // Number of points in eta  + 2
  int ncy,                // Number of points in chi1 + 2
  int ncz,                // Number of points in chi2 + 2
  const double *etavec,   // B-spline knots in eta
  const double *chi1vec,  // B-spline knots in chi1
  const double *chi2vec   // B-spline knots in chi2
)
{
  if(!splinedata) exit(1);
  if(*splinedata) SplineData_Destroy(*splinedata);

  (*splinedata)=XLALCalloc(1,sizeof(SplineData));

  // Set up B-spline basis for desired knots
  const size_t nbreak_x = ncx-2;  // must have nbreak = n-2 for cubic splines
  const size_t nbreak_y = ncy-2;  // must have nbreak = n-2 for cubic splines
  const size_t nbreak_z = ncz-2;  // must have nbreak = n-2 for cubic splines

  // Allocate a cubic bspline workspace (k = 4)
  gsl_bspline_workspace *bwx = gsl_bspline_alloc(4, nbreak_x);
  gsl_bspline_workspace *bwy = gsl_bspline_alloc(4, nbreak_y);
  gsl_bspline_workspace *bwz = gsl_bspline_alloc(4, nbreak_z);

  // Set breakpoints (and thus knots by hand)
  gsl_vector *breakpts_x = gsl_vector_alloc(nbreak_x);
  gsl_vector *breakpts_y = gsl_vector_alloc(nbreak_y);
  gsl_vector *breakpts_z = gsl_vector_alloc(nbreak_z);
  for (UINT4 i=0; i<nbreak_x; i++)
    gsl_vector_set(breakpts_x, i, etavec[i]);
  for (UINT4 j=0; j<nbreak_y; j++)
    gsl_vector_set(breakpts_y, j, chi1vec[j]);
  for (UINT4 k=0; k<nbreak_z; k++)
    gsl_vector_set(breakpts_z, k, chi2vec[k]);

  gsl_bspline_knots(breakpts_x, bwx);
  gsl_bspline_knots(breakpts_y, bwy);
  gsl_bspline_knots(breakpts_z, bwz);

  gsl_vector_free(breakpts_x);
  gsl_vector_free(breakpts_y);
  gsl_vector_free(breakpts_z);

  (*splinedata)->bwx=bwx;
  (*splinedata)->bwy=bwy;
  (*splinedata)->bwz=bwz;
}

static void SplineData_Destroy(SplineData *splinedata)
{
  if(!splinedata) return;
  if(splinedata->bwx) gsl_bspline_free(splinedata->bwx);
  if(splinedata->bwy) gsl_bspline_free(splinedata->bwy);
  if(splinedata->bwz) gsl_bspline_free(splinedata->bwz);
  XLALFree(splinedata);
}

// Interpolate projection coefficients for amplitude and phase over the parameter space (q, chi).
// The multi-dimensional interpolation is carried out via a tensor product decomposition.
static int TP_Spline_interpolation_3d(
  REAL8 eta,                // Input: eta-value for which projection coefficients should be evaluated
  REAL8 chi1,               // Input: chi1-value for which projection coefficients should be evaluated
  REAL8 chi2,               // Input: chi2-value for which projection coefficients should be evaluated
  gsl_vector *cvec_amp,     // Input: data for spline coefficients for amplitude
  gsl_vector *cvec_phi,     // Input: data for spline coefficients for phase
  int nk_amp,               // number of SVD-modes == number of basis functions for amplitude
  int nk_phi,               // number of SVD-modes == number of basis functions for phase
  int nk_max,               // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
  int ncx,                  // Number of points in eta  + 2
  int ncy,                  // Number of points in chi1 + 2
  int ncz,                  // Number of points in chi2 + 2
  const double *etavec,     // B-spline knots in eta
  const double *chi1vec,    // B-spline knots in chi1
  const double *chi2vec,    // B-spline knots in chi2
  gsl_vector *c_amp,        // Output: interpolated projection coefficients for amplitude
  gsl_vector *c_phi         // Output: interpolated projection coefficients for phase
  ) {
  if (nk_max != -1) {
    if (nk_max > nk_amp || nk_max > nk_phi)
      XLAL_ERROR(XLAL_EDOM, "Truncation parameter nk_max %d must be smaller or equal to nk_amp %d and nk_phi %d", nk_max, nk_amp, nk_phi);
    else { // truncate SVD modes
      nk_amp = nk_max;
      nk_phi = nk_max;
    }
  }

  SplineData *splinedata=NULL;
  SplineData_Init(&splinedata, ncx, ncy, ncz, etavec, chi1vec, chi2vec);

  gsl_bspline_workspace *bwx=splinedata->bwx;
  gsl_bspline_workspace *bwy=splinedata->bwy;
  gsl_bspline_workspace *bwz=splinedata->bwz;

  int N = ncx*ncy*ncz;  // Size of the data matrix for one SVD-mode
  // Evaluate the TP spline for all SVD modes - amplitude
  for (int k=0; k<nk_amp; k++) { // For each SVD mode
    gsl_vector v = gsl_vector_subvector(cvec_amp, k*N, N).vector; // Pick out the coefficient matrix corresponding to the k-th SVD mode.
    REAL8 csum = Interpolate_Coefficent_Tensor(&v, eta, chi1, chi2, ncy, ncz, bwx, bwy, bwz);
    gsl_vector_set(c_amp, k, csum);
  }

  // Evaluate the TP spline for all SVD modes - phase
  for (int k=0; k<nk_phi; k++) {  // For each SVD mode
    gsl_vector v = gsl_vector_subvector(cvec_phi, k*N, N).vector; // Pick out the coefficient matrix corresponding to the k-th SVD mode.
    REAL8 csum = Interpolate_Coefficent_Tensor(&v, eta, chi1, chi2, ncy, ncz, bwx, bwy, bwz);
    gsl_vector_set(c_phi, k, csum);
  }

  SplineData_Destroy(splinedata);

  return(0);
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
  LALH5File *sub = XLALH5GroupOpen(file, grp_name);

  // Read ROM coefficients
  ReadHDF5RealVectorDataset(sub, "Amp_ciall", & (*submodel)->cvec_amp);
  ReadHDF5RealVectorDataset(sub, "Phase_ciall", & (*submodel)->cvec_phi);

  // Read ROM basis functions
  ReadHDF5RealMatrixDataset(sub, "Bamp", & (*submodel)->Bamp);
  ReadHDF5RealMatrixDataset(sub, "Bphase", & (*submodel)->Bphi);

  // Read sparse frequency points
  ReadHDF5RealVectorDataset(sub, "Mf_grid_Amp", & (*submodel)->gA);
  ReadHDF5RealVectorDataset(sub, "Mf_grid_Phi", & (*submodel)->gPhi);

  // Read parameter space nodes
  ReadHDF5RealVectorDataset(sub, "etavec", & (*submodel)->etavec);
  ReadHDF5RealVectorDataset(sub, "chi1vec", & (*submodel)->chi1vec);
  ReadHDF5RealVectorDataset(sub, "chi2vec", & (*submodel)->chi2vec);

  // Initialize other members
  (*submodel)->nk_amp = (*submodel)->gA->size;
  (*submodel)->nk_phi = (*submodel)->gPhi->size;
  (*submodel)->ncx = (*submodel)->etavec->size + 2;
  (*submodel)->ncy = (*submodel)->chi1vec->size + 2;
  (*submodel)->ncz = (*submodel)->chi2vec->size + 2;

  // Domain of definition of submodel
  (*submodel)->eta_bounds[0] = gsl_vector_get((*submodel)->etavec, 0);
  (*submodel)->eta_bounds[1] = gsl_vector_get((*submodel)->etavec, (*submodel)->etavec->size - 1);
  (*submodel)->chi1_bounds[0] = gsl_vector_get((*submodel)->chi1vec, 0);
  (*submodel)->chi1_bounds[1] = gsl_vector_get((*submodel)->chi1vec, (*submodel)->chi1vec->size - 1);
  (*submodel)->chi2_bounds[0] = gsl_vector_get((*submodel)->chi2vec, 0);
  (*submodel)->chi2_bounds[1] = gsl_vector_get((*submodel)->chi2vec, (*submodel)->chi2vec->size - 1);

  XLALFree(path);
  XLALH5FileClose(file);

  // NEW CODE FOR testing TEOBv4


  LALH5File *file2 = XLALH5FileOpen("/Users/mpuer/Documents/gpsurrogate/src/TEOB-LAL-implementation/TEOBv4_surrogate.hdf5", "r");
  LALH5File *root = XLALH5GroupOpen(file2, "/"); // most convenient to open the root once

  //////////////////////////////////////////////////////////////////////////////
  // load everything we need
  // GP hyperparameters
  ReadHDF5RealMatrixDataset(root, "hyp_amp", & (*submodel)->hyp_amp);
  ReadHDF5RealMatrixDataset(root, "hyp_phi", & (*submodel)->hyp_phi);

  // kinv_dot_y
  ReadHDF5RealMatrixDataset(root, "kinv_dot_y_amp", & (*submodel)->kinv_dot_y_amp);
  ReadHDF5RealMatrixDataset(root, "kinv_dot_y_phi", & (*submodel)->kinv_dot_y_phi);

  // Reduced bases
  // ReadHDF5RealMatrixDataset(root, "B_amp", & (*submodel)->B_amp);
  // ReadHDF5RealMatrixDataset(root, "B_phi", & (*submodel)->B_phi);

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

  // cleanup

  XLALH5FileClose(file2);

  // END NEW CODE

  ret = XLAL_SUCCESS;
#else
  XLAL_ERROR(XLAL_EFAILED, "HDF5 support not enabled");
#endif

  return ret;
}

/* Deallocate contents of the given TEOBv4ROMdataDS_submodel structure */
static void TEOBv4ROMdataDS_Cleanup_submodel(TEOBv4ROMdataDS_submodel *submodel) {
  // FIXME: remove these
  if(submodel->cvec_amp) gsl_vector_free(submodel->cvec_amp);
  if(submodel->cvec_phi) gsl_vector_free(submodel->cvec_phi);
  if(submodel->Bamp) gsl_matrix_free(submodel->Bamp);
  if(submodel->Bphi) gsl_matrix_free(submodel->Bphi);
  if(submodel->gA)   gsl_vector_free(submodel->gA);
  if(submodel->gPhi) gsl_vector_free(submodel->gPhi);
  if(submodel->etavec)  gsl_vector_free(submodel->etavec);
  if(submodel->chi1vec) gsl_vector_free(submodel->chi1vec);
  if(submodel->chi2vec) gsl_vector_free(submodel->chi2vec);

  if(submodel->hyp_amp) gsl_matrix_free(submodel->hyp_amp);
  if(submodel->hyp_phi) gsl_matrix_free(submodel->hyp_phi);
  if(submodel->kinv_dot_y_amp) gsl_matrix_free(submodel->kinv_dot_y_amp);
  if(submodel->kinv_dot_y_phi) gsl_matrix_free(submodel->kinv_dot_y_phi);
  // if(submodel->B_amp) gsl_matrix_free(submodel->B_amp);
  // if(submodel->B_phi) gsl_matrix_free(submodel->B_phi);

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
fprintf(stderr, "In TEOBv4ROMdataDS_Init()\n");
#ifdef LAL_HDF5_ENABLED
  // First, check we got the correct version number
  size_t size = strlen(dir) + strlen(ROMDataHDF5) + 2;
  char *path = XLALMalloc(size);
  snprintf(path, size, "%s/%s", dir, ROMDataHDF5);
  LALH5File *file = XLALH5FileOpen(path, "r");

  XLALPrintInfo("ROM metadata\n============\n");
  PrintInfoStringAttribute(file, "Email");
  PrintInfoStringAttribute(file, "Description");
  ret = ROM_check_version_number(file, ROMDataHDF5_VERSION_MAJOR,
                                 ROMDataHDF5_VERSION_MINOR,
                                 ROMDataHDF5_VERSION_MICRO);

  XLALFree(path);
  XLALH5FileClose(file);
fprintf(stderr, "In TEOBv4ROMdataDS_Init(): read metadata\n");
  ret |= TEOBv4ROMdataDS_Init_submodel(&(romdata)->sub1, dir, "sub1");
  if (ret==XLAL_SUCCESS) XLALPrintInfo("%s : submodel 1 loaded successfully.\n", __func__);

  ret |= TEOBv4ROMdataDS_Init_submodel(&(romdata)->sub2, dir, "sub2");
  if (ret==XLAL_SUCCESS) XLALPrintInfo("%s : submodel 2 loaded successfully.\n", __func__);

  ret |= TEOBv4ROMdataDS_Init_submodel(&(romdata)->sub3, dir, "sub3");
  if (ret==XLAL_SUCCESS) XLALPrintInfo("%s : submodel 3 loaded successfully.\n", __func__);
fprintf(stderr, "In TEOBv4ROMdataDS_Init(): read submodels: %d\n", ret);
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
  TEOBv4ROMdataDS_Cleanup_submodel((romdata)->sub2);
  XLALFree((romdata)->sub2);
  (romdata)->sub2 = NULL;
  TEOBv4ROMdataDS_Cleanup_submodel((romdata)->sub3);
  XLALFree((romdata)->sub3);
  (romdata)->sub3 = NULL;
  romdata->setup=0;
}

/* Structure for internal use */
static void TEOBv4ROMdataDS_coeff_Init(TEOBv4ROMdataDS_coeff **romdatacoeff, int nk_amp, int nk_phi) {
  if(!romdatacoeff) exit(1);
  /* Create storage for structures */
  if(!*romdatacoeff)
    *romdatacoeff=XLALCalloc(1,sizeof(TEOBv4ROMdataDS_coeff));
  else
    TEOBv4ROMdataDS_coeff_Cleanup(*romdatacoeff);

  (*romdatacoeff)->c_amp = gsl_vector_alloc(nk_amp);
  (*romdatacoeff)->c_phi = gsl_vector_alloc(nk_phi);
}

/* Deallocate contents of the given TEOBv4ROMdataDS_coeff structure */
static void TEOBv4ROMdataDS_coeff_Cleanup(TEOBv4ROMdataDS_coeff *romdatacoeff) {
  if(romdatacoeff->c_amp) gsl_vector_free(romdatacoeff->c_amp);
  if(romdatacoeff->c_phi) gsl_vector_free(romdatacoeff->c_phi);
  XLALFree(romdatacoeff);
}

/* Return the closest higher power of 2  */
// Note: NextPow(2^k) = 2^k for integer values k.
static size_t NextPow2(const size_t n) {
  return 1 << (size_t) ceil(log2(n));
}

static void GlueAmplitude(
  // INPUTS
  TEOBv4ROMdataDS_submodel *submodel_lo,
  TEOBv4ROMdataDS_submodel *submodel_hi,
  gsl_vector* amp_f_lo,
  gsl_vector* amp_f_hi,
  // amp_pre_* can be set to 1 if not using amplitude prefactors
  double amp_pre_lo,
  double amp_pre_hi,
  const double Mfm,
  // OUTPUTS
  gsl_interp_accel **acc_amp,
  gsl_spline **spline_amp
) {
  // First need to find overlaping frequency interval
  int jA_lo;
  // Find index so that Mf < Mfm
  for (jA_lo=0; jA_lo < submodel_lo->nk_amp; jA_lo++) {
    if (gsl_vector_get(submodel_lo->gA, jA_lo) > Mfm) {
      jA_lo--;
      break;
    }
  }

  int jA_hi;
  // Find index so that Mf > Mfm
  for (jA_hi=0; jA_hi < submodel_hi->nk_amp; jA_hi++)
    if (gsl_vector_get(submodel_hi->gA, jA_hi) > Mfm)
      break;

  int nA = 1 + jA_lo + (submodel_hi->nk_amp - jA_hi); // length of the union of frequency points of the low and high frequency models glued at MfM

  gsl_vector *gAU = gsl_vector_alloc(nA); // glued frequency grid
  gsl_vector *amp_f = gsl_vector_alloc(nA); // amplitude on glued frequency grid
  // Note: We don't interpolate the amplitude, but this may already be smooth enough for practical purposes.
  // To improve this we would evaluate both amplitue splines times the prefactor at the matching frequency and correct with the ratio, so we are C^0.
  for (int i=0; i<=jA_lo; i++) {
    gsl_vector_set(gAU, i, gsl_vector_get(submodel_lo->gA, i));
    double A = amp_pre_lo * gsl_vector_get(amp_f_lo, i);
    gsl_vector_set(amp_f, i, A);
  }

  for (int i=jA_lo+1; i<nA; i++) {
    int k = jA_hi - (jA_lo+1) + i;
    gsl_vector_set(gAU, i, gsl_vector_get(submodel_hi->gA, k));
    double A = amp_pre_hi * gsl_vector_get(amp_f_hi, k);
    gsl_vector_set(amp_f, i, A);
  }

  // Setup 1d splines in frequency from glued amplitude grids & data
  *acc_amp = gsl_interp_accel_alloc();
  *spline_amp = gsl_spline_alloc(gsl_interp_cspline, nA);
  //gsl_spline_init(spline_amp, gAU->data, gsl_vector_const_ptr(amp_f,0), nA);
  gsl_spline_init(*spline_amp, gsl_vector_const_ptr(gAU,0),
                  gsl_vector_const_ptr(amp_f,0), nA);

  gsl_vector_free(gAU);
  gsl_vector_free(amp_f);
  gsl_vector_free(amp_f_lo);
  gsl_vector_free(amp_f_hi);
}

// Glue phasing in frequency to C^1 smoothness
static void GluePhasing(
  // INPUTS
  TEOBv4ROMdataDS_submodel *submodel_lo,
  TEOBv4ROMdataDS_submodel *submodel_hi,
  gsl_vector* phi_f_lo,
  gsl_vector* phi_f_hi,
  const double Mfm,
  // OUTPUTS
  gsl_interp_accel **acc_phi,
  gsl_spline **spline_phi
) {
  // First need to find overlaping frequency interval
  int jP_lo;
  // Find index so that Mf < Mfm
  for (jP_lo=0; jP_lo < submodel_lo->nk_phi; jP_lo++) {
    if (gsl_vector_get(submodel_lo->gPhi, jP_lo) > Mfm) {
      jP_lo--;
      break;
    }
  }

  int jP_hi;
  // Find index so that Mf > Mfm
  for (jP_hi=0; jP_hi < submodel_hi->nk_phi; jP_hi++)
    if (gsl_vector_get(submodel_hi->gPhi, jP_hi) > Mfm)
      break;

  int nP = 1 + jP_lo + (submodel_hi->nk_phi - jP_hi); // length of the union of frequency points of the low and high frequency models glued at MfM
  gsl_vector *gPU = gsl_vector_alloc(nP); // glued frequency grid
  gsl_vector *phi_f = gsl_vector_alloc(nP); // phase on glued frequency grid
  // We need to do a bit more work to glue the phase with C^1 smoothness
  for (int i=0; i<=jP_lo; i++) {
    gsl_vector_set(gPU, i, gsl_vector_get(submodel_lo->gPhi, i));
    double P = gsl_vector_get(phi_f_lo, i);
    gsl_vector_set(phi_f, i, P);
  }

  for (int i=jP_lo+1; i<nP; i++) {
    int k = jP_hi - (jP_lo+1) + i;
    gsl_vector_set(gPU, i, gsl_vector_get(submodel_hi->gPhi, k));
    double P = gsl_vector_get(phi_f_hi, k);
    gsl_vector_set(phi_f, i, P);
  }

  // Set up phase data across the gluing frequency Mfm
  // We need to set up a spline for the low frequency model and evaluate at the designated points for the high frequency model so that we work with the *same* frequeny interval!

  // We could optimize this further by not constructing the whole spline for
  // submodel_lo, but this may be insignificant since the number of points is small anyway.
  gsl_interp_accel *acc_phi_lo = gsl_interp_accel_alloc();
  gsl_spline *spline_phi_lo = gsl_spline_alloc(gsl_interp_cspline, submodel_lo->nk_phi);
  gsl_spline_init(spline_phi_lo, gsl_vector_const_ptr(submodel_lo->gPhi,0),
                  gsl_vector_const_ptr(phi_f_lo,0), submodel_lo->nk_phi);

  const int nn = 15;
  gsl_vector_const_view gP_hi_data = gsl_vector_const_subvector(submodel_hi->gPhi, jP_hi - nn, 2*nn+1);
  gsl_vector_const_view P_hi_data = gsl_vector_const_subvector(phi_f_hi, jP_hi - nn, 2*nn+1);
  gsl_vector *P_lo_data = gsl_vector_alloc(2*nn+1);
  for (int i=0; i<2*nn+1; i++) {
    double P = gsl_spline_eval(spline_phi_lo, gsl_vector_get(&gP_hi_data.vector, i), acc_phi_lo);
    gsl_vector_set(P_lo_data, i, P);
  }

  // Fit phase data to cubic polynomial in frequency
  gsl_vector *cP_lo = Fit_cubic(&gP_hi_data.vector, P_lo_data);
  gsl_vector *cP_hi = Fit_cubic(&gP_hi_data.vector, &P_hi_data.vector);

  double P_lo_derivs[2];
  double P_hi_derivs[2];
  gsl_poly_eval_derivs(cP_lo->data, 4, Mfm, P_lo_derivs, 2);
  gsl_poly_eval_derivs(cP_hi->data, 4, Mfm, P_hi_derivs, 2);

  double delta_omega = P_hi_derivs[1] - P_lo_derivs[1];
  double delta_phi   = P_hi_derivs[0] - P_lo_derivs[0] - delta_omega * Mfm;

  for (int i=jP_lo+1; i<nP; i++) {
    int k = jP_hi - (jP_lo+1) + i;
    double f = gsl_vector_get(submodel_hi->gPhi, k);
    gsl_vector_set(gPU, i, f);
    double P = gsl_vector_get(phi_f_hi, k) - delta_omega * f - delta_phi; // Now correct phase of high frequency submodel
    gsl_vector_set(phi_f, i, P);
  }

  // free some vectors
  gsl_vector_free(P_lo_data);
  gsl_vector_free(cP_lo);
  gsl_vector_free(cP_hi);
  gsl_vector_free(phi_f_lo);
  gsl_vector_free(phi_f_hi);

  // Setup 1d splines in frequency from glued phase grids & data
  *acc_phi = gsl_interp_accel_alloc();
  *spline_phi = gsl_spline_alloc(gsl_interp_cspline, nP);
  //gsl_spline_init(spline_phi, gPU->data, gsl_vector_const_ptr(phi_f,0), nP);
  gsl_spline_init(*spline_phi, gsl_vector_const_ptr(gPU,0),
                  gsl_vector_const_ptr(phi_f,0), nP);

  /**** Finished gluing ****/

  gsl_vector_free(phi_f);
  gsl_vector_free(gPU);
  gsl_spline_free(spline_phi_lo);
  gsl_interp_accel_free(acc_phi_lo);
}

static int TaylorF2Phasing(
  double Mtot,
  double q,
  double chi1,
  double chi2,
  double lambda1,
  double lambda2,
  gsl_vector *Mf
);

static int TaylorF2Phasing(
  double Mtot,
  double q,
  double chi1,
  double chi2,
  double lambda1,
  double lambda2,
  gsl_vector *Mfs
) {
  PNPhasingSeries *pn = NULL;
  LALDict *extraParams = XLALCreateDict();
  XLALSimInspiralWaveformParamsInsertPNSpinOrder(extraParams, LAL_SIM_INSPIRAL_SPIN_ORDER_35PN);
  // Ideally, we should be able to specify lambdas like this, but for now the phasing functon does not use the lambda parameters!
  // XLALSimInspiralWaveformParamsInsertTidalLambda1(extraParams, lambda1);
  // XLALSimInspiralWaveformParamsInsertTidalLambda2(extraParams, lambda2);
  // See src/LALSimInspiralTaylorF2.c:XLALSimInspiralTaylorF2AlignedPhasingArray() for how to add the tidal terms
  // We add the terms after the XLALSimInspiralTaylorF2AlignedPhasing() call below to the structure of PN coefficients.

  // FIXME: We should be able to switch on quadrupole-monopole terms (self-spin deformation of the two bodies)
  // from TaylorF2; later these can be replaced with EOB as well. Just add a switch for now.
  // XLALSimInspiralWaveformParamsInsertdQuadMon1(extraParams, dquadmon1);
  // XLALSimInspiralWaveformParamsInsertdQuadMon2(extraParams, dquadmon2);

  double m1OverM = q / (1.0+q);
  double m2OverM = 1.0 / (1.0+q);
  double m1 = Mtot * m1OverM * LAL_MSUN_SI;
  double m2 = Mtot * m2OverM * LAL_MSUN_SI;
  XLALSimInspiralTaylorF2AlignedPhasing(&pn, m1, m2, chi1, chi2, extraParams);

fprintf(stderr, "%g %g %g %g %g\n", Mtot, m1, m2, chi1, chi2);

  // Add tidal deformability terms
  pn->v[10] = pn->v[0] * ( lambda1 * XLALSimInspiralTaylorF2Phasing_10PNTidalCoeff(m1OverM)
                         + lambda2 * XLALSimInspiralTaylorF2Phasing_10PNTidalCoeff(m2OverM) );
  pn->v[12] = pn->v[0] * ( lambda1 * XLALSimInspiralTaylorF2Phasing_12PNTidalCoeff(m1OverM)
                         + lambda2 * XLALSimInspiralTaylorF2Phasing_12PNTidalCoeff(m2OverM) );

  // Total amplitude and phase (PN with added surrogate corrections)
  //gsl_vector *amp_at_nodes = gsl_vector_alloc(N_amp);
  //gsl_vector *phi_at_nodes = gsl_vector_alloc(N_phi);

fprintf(stderr, "pn->v: %g %g %g %g %g %g %g %g \n", pn->v[7], pn->v[6], pn->v[5], pn->v[4], pn->v[3], pn->v[2], pn->v[1], pn->v[0]);
fprintf(stderr, "pn->vlogv: %g %g\n", pn->vlogv[6], pn->vlogv[5]);

fprintf(stderr, "\nPN phasing at nodes:");
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
      // shft = LAL_TWOPI * (tC.gpsSeconds + 1e-9 * tC.gpsNanoSeconds);
      // phasing += shft * f - 2.*phi_ref - ref_phasing; // FIXME: add ref

      fprintf(stderr, "%.15g\n", phasing);

      //fprintf(stderr, "PN phasing[%zu] = %g, %g\n", i, Mf, phasing);
      //gsl_vector_set(phi_at_nodes, i, phasing + gsl_vector_get(sur_phi_at_nodes, i));
      // TODO: compare total and all the terms
  }
  fprintf(stderr, "\n");
  //amp * cos(phasing - LAL_PI_4) - amp * sin(phasing - LAL_PI_4) * 1.0j;
 // amp0 = -4. * m1 * m2 / r * LAL_MRSUN_SI * LAL_MTSUN_SI * sqrt(LAL_PI/12.L);

  XLALDestroyDict(extraParams);
  XLALFree(pn);

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
  double deltaF,
  /* If deltaF > 0, the frequency points given in freqs are uniformly spaced with
   * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
   * Then we will use deltaF = 0 to create the frequency series we return. */
  int nk_max // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
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

  /* We always need to glue two submodels together for this ROM */
  TEOBv4ROMdataDS_submodel *submodel_hi; // high frequency ROM
  TEOBv4ROMdataDS_submodel *submodel_lo; // low frequency ROM
  submodel_lo = romdata->sub1;

  /* Select high frequency ROM submodel */
  if (chi1 < romdata->sub3->chi1_bounds[0] || eta > romdata->sub3->eta_bounds[1]) // only check the two conditions that apply for this ROM; could be more general, but slower
    submodel_hi = romdata->sub2;
  else
    submodel_hi = romdata->sub3;


  /* Find frequency bounds */
  if (!freqs_in) XLAL_ERROR(XLAL_EFAULT);
  double fLow  = freqs_in->data[0];
  double fHigh = freqs_in->data[freqs_in->length - 1];

  if(fRef==0.0)
    fRef=fLow;

  /* Convert to geometric units for frequency */
  // lowest allowed geometric frequency for ROM
  double Mf_ROM_min = fmax(gsl_vector_get(submodel_lo->gA, 0),
                           gsl_vector_get(submodel_lo->gPhi,0));
  // highest allowed geometric frequency for ROM
  double Mf_ROM_max = fmin(gsl_vector_get(submodel_hi->gA, submodel_hi->nk_amp-1),
                           gsl_vector_get(submodel_hi->gPhi, submodel_hi->nk_phi-1));
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

  /* Internal storage for waveform coefficiencts */
  TEOBv4ROMdataDS_coeff *romdata_coeff_lo=NULL;
  TEOBv4ROMdataDS_coeff *romdata_coeff_hi=NULL;
  TEOBv4ROMdataDS_coeff_Init(&romdata_coeff_lo, submodel_lo->nk_amp, submodel_lo->nk_phi);
  TEOBv4ROMdataDS_coeff_Init(&romdata_coeff_hi, submodel_hi->nk_amp, submodel_hi->nk_phi);
  REAL8 amp_pre_lo = 1.0; // unused here
  REAL8 amp_pre_hi = 1.0;


  // TODO: Call GPR instead for log amplitude and dephasing
  double q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  int N_amp = submodel_lo->mf_amp->size;
  int N_phi = submodel_lo->mf_phi->size; // should already be corrected
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




  /* Interpolate projection coefficients and evaluate them at (eta,chi1,chi2) */
  retcode=TP_Spline_interpolation_3d(
    eta,                          // Input: eta-value for which projection coefficients should be evaluated
    chi1,                         // Input: chi1-value for which projection coefficients should be evaluated
    chi2,                         // Input: chi2-value for which projection coefficients should be evaluated
    submodel_lo->cvec_amp,        // Input: data for spline coefficients for amplitude
    submodel_lo->cvec_phi,        // Input: data for spline coefficients for phase
    submodel_lo->nk_amp,          // number of SVD-modes == number of basis functions for amplitude
    submodel_lo->nk_phi,          // number of SVD-modes == number of basis functions for phase
    nk_max,                       // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
    submodel_lo->ncx,             // Number of points in eta  + 2
    submodel_lo->ncy,             // Number of points in chi1 + 2
    submodel_lo->ncz,             // Number of points in chi2 + 2
    gsl_vector_const_ptr(submodel_lo->etavec, 0),          // B-spline knots in eta
    gsl_vector_const_ptr(submodel_lo->chi1vec, 0),        // B-spline knots in chi1
    gsl_vector_const_ptr(submodel_lo->chi2vec, 0),        // B-spline knots in chi2
    romdata_coeff_lo->c_amp,      // Output: interpolated projection coefficients for amplitude
    romdata_coeff_lo->c_phi       // Output: interpolated projection coefficients for phase
  );

  if(retcode!=0) {
    TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_lo);
    XLAL_ERROR(retcode);
  }

  /* Interpolate projection coefficients and evaluate them at (eta,chi1,chi2) */
  retcode=TP_Spline_interpolation_3d(
    eta,                          // Input: eta-value for which projection coefficients should be evaluated
    chi1,                         // Input: chi1-value for which projection coefficients should be evaluated
    chi2,                         // Input: chi2-value for which projection coefficients should be evaluated
    submodel_hi->cvec_amp,        // Input: data for spline coefficients for amplitude
    submodel_hi->cvec_phi,        // Input: data for spline coefficients for phase
    submodel_hi->nk_amp,          // number of SVD-modes == number of basis functions for amplitude
    submodel_hi->nk_phi,          // number of SVD-modes == number of basis functions for phase
    nk_max,                       // truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1
    submodel_hi->ncx,             // Number of points in eta  + 2
    submodel_hi->ncy,             // Number of points in chi1 + 2
    submodel_hi->ncz,             // Number of points in chi2 + 2
    gsl_vector_const_ptr(submodel_hi->etavec, 0),         // B-spline knots in eta
    gsl_vector_const_ptr(submodel_hi->chi1vec, 0),        // B-spline knots in chi1
    gsl_vector_const_ptr(submodel_hi->chi2vec, 0),        // B-spline knots in chi2
    romdata_coeff_hi->c_amp,      // Output: interpolated projection coefficients for amplitude
    romdata_coeff_hi->c_phi       // Output: interpolated projection coefficients for phase
  );

  if(retcode!=0) {
    TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_hi);
    XLAL_ERROR(retcode);
  }


  // Compute function values of amplitude an phase on sparse frequency points by evaluating matrix vector products
  // amp_pts = B_A^T . c_A
  // phi_pts = B_phi^T . c_phi
  gsl_vector* amp_f_lo = gsl_vector_alloc(submodel_lo->nk_amp);
  gsl_vector* phi_f_lo = gsl_vector_alloc(submodel_lo->nk_phi);
  gsl_blas_dgemv(CblasTrans, 1.0, submodel_lo->Bamp, romdata_coeff_lo->c_amp, 0.0, amp_f_lo);
  gsl_blas_dgemv(CblasTrans, 1.0, submodel_lo->Bphi, romdata_coeff_lo->c_phi, 0.0, phi_f_lo);

  gsl_vector* amp_f_hi = gsl_vector_alloc(submodel_hi->nk_amp);
  gsl_vector* phi_f_hi = gsl_vector_alloc(submodel_hi->nk_phi);
  gsl_blas_dgemv(CblasTrans, 1.0, submodel_hi->Bamp, romdata_coeff_hi->c_amp, 0.0, amp_f_hi);
  gsl_blas_dgemv(CblasTrans, 1.0, submodel_hi->Bphi, romdata_coeff_hi->c_phi, 0.0, phi_f_hi);

  const double Mfm = 0.01; // Gluing frequency: the low and high frequency ROMs overlap here; this is used both for amplitude and phase.



  // Evaluate ln amplitude and dephasing
  // int len = (submodel_lo->B_amp)->size2; // FIXME: use a constant instead
  // gsl_vector* amp_f = gsl_vector_alloc(len);
  // gsl_vector* phi_f = gsl_vector_alloc(len);
  // // THIS is no longer needed
  // gsl_blas_dgemv(CblasTrans, 1.0, submodel_lo->B_amp, amp_at_EI_nodes, 0.0, amp_f);
  // gsl_blas_dgemv(CblasTrans, 1.0, submodel_lo->B_phi, phi_at_EI_nodes, 0.0, phi_f);

  fprintf(stderr, "\namp_at_nodes:");
  gsl_vector_fprintf(stderr, sur_amp_at_nodes, "%.15g");
  fprintf(stderr, "\nphi_at_nodes:");
  gsl_vector_fprintf(stderr, sur_phi_at_nodes, "%.15g");

// FIXME:
//   * evaluate TF2 basemodel on same frequencies
//   * add amplitude and phase correction
//   * then spline them (is that OK, or do we have to spline the correction and TF2 first??)

  // TODO: evaluate hardcoded TF2 and reconstruct waveform
  // fprintf(stderr, "%zu %zu %zu\n", amp_at_EI_nodes->size, (submodel_lo->B_amp)->size1, (submodel_lo->B_amp)->size2);

TaylorF2Phasing(Mtot, q, chi1, chi2, lambda1, lambda2, submodel_lo->mf_phi);

  // TODO: splines for amp, phi
  // FIXME: deallocate all amp, phi variables: PN, surrogate corrections

  // // See LALSimInspiralPNCoefficients.c:XLALSimInspiralPNPhasing_F2()
  // pn->v[5], ....

  // or get complete TF2 waveform
  // FIXME: check units: f, m
  // FIXME: destroy hptilde_TF2 after using it or use hptilde variable
  // double m1_SI = Mtot * q/(1.0+q) * LAL_MSUN_SI;
  // double m2_SI = Mtot * 1.0/(1.0+q) * LAL_MSUN_SI;
  // // COMPLEX16FrequencySeries **hptilde_TF2 = NULL;
  // LALDict *LALparams = NULL;
  // XLALSimInspiralTaylorF2(hptilde_TF2, phiRef, deltaF, m1_SI, m2_SI, chi1, chi2, fLow, fHigh, fRef,
  // distance, LALparams);

  // phasing
  // FIXME: call XLALSimInspiralTaylorF2AlignedPhasing() to get the point-particle phasing in a structure


  // FIXME: reimplement these functions since they are static; better: make them XLAL
  // static REAL8 UNUSED
  // XLALSimInspiralTaylorF2Phasing_10PNTidalCoeff(
  // 	REAL8 mByM /**< ratio of object mass to total mass */
  //     )
  // {
  //   return (-288. + 264.*mByM)*mByM*mByM*mByM*mByM;
  //
  // }
  //
  // static REAL8 UNUSED
  // XLALSimInspiralTaylorF2Phasing_12PNTidalCoeff(
  // 	REAL8 mByM /**< ratio of object mass to total mass */
  //     )
  // {
  //   return (-15895./28. + 4595./28.*mByM + 5715./14.*mByM*mByM - 325./7.*mByM*mByM*mByM)*mByM*mByM*mByM*mByM;
  // }

  // implement this as a separate function
  // amplitude taylorf2_amp()
  // https://github.com/benjaminlackey/gpsurrogate/blob/master/src/taylorf2.py#L59


  // Glue amplitude
  gsl_interp_accel *acc_amp;
  gsl_spline *spline_amp;
  GlueAmplitude(submodel_lo, submodel_hi, amp_f_lo, amp_f_hi, amp_pre_lo, amp_pre_hi, Mfm,
    &acc_amp, &spline_amp
  );

  // Glue phasing in frequency to C^1 smoothness
  gsl_interp_accel *acc_phi;
  gsl_spline *spline_phi;
  GluePhasing(submodel_lo, submodel_hi, phi_f_lo, phi_f_hi, Mfm,
    &acc_phi, &spline_phi
  );

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
      TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_lo);
      TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_hi);
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

  // Assemble waveform from aplitude and phase
  for (UINT4 i=0; i<freqs->length; i++) { // loop over frequency points in sequence
    double f = freqs->data[i];
    if (f > Mf_ROM_max) continue; // We're beyond the highest allowed frequency; since freqs may not be ordered, we'll just skip the current frequency and leave zero in the buffer
    int j = i + offset; // shift index for frequency series if needed
    double A = gsl_spline_eval(spline_amp, f, acc_amp);
    double phase = gsl_spline_eval(spline_phi, f, acc_phi) - phase_change;
    COMPLEX16 htilde = s*amp0*A * (cos(phase) + I*sin(phase));//cexp(I*phase);
    pdata[j] =      pcoef * htilde;
    cdata[j] = -I * ccoef * htilde;
  }

  /* Correct phasing so we coalesce at t=0 (with the definition of the epoch=-1/deltaF above) */

  // Get SEOBNRv4 ringdown frequency for 22 mode
  double Mf_final = SEOBNRROM_Ringdown_Mf_From_Mtot_Eta(Mtot_sec, eta, chi1,
                                                        chi2, SEOBNRv4);

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
    TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_lo);
    TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_hi);
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

  gsl_spline_free(spline_amp);
  gsl_spline_free(spline_phi);
  gsl_interp_accel_free(acc_amp);
  gsl_interp_accel_free(acc_phi);
  TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_lo);
  TEOBv4ROMdataDS_coeff_Cleanup(romdata_coeff_hi);

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
 *   * 2Msun (@ flow=20Hz) <= Mtot < 500Msun
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
  REAL8 lambda2,                                /**< Dimensionless tidal deformability 2 */
  INT4 nk_max)                                  /**< Truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1 */
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
                                inclination, Mtot_sec, eta, chi1, chi2, lambda1, lambda2, freqs,
                                0, nk_max);

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
  REAL8 lambda2,                                /**< Dimensionless tidal deformability 2 */
  INT4 nk_max)                                  /**< Truncate interpolants at SVD mode nk_max; don't truncate if nk_max == -1 */
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
                                inclination, Mtot_sec, eta, chi1, chi2, lambda1, lambda2, freqs,
                                deltaF, nk_max);

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
