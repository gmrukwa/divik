/*
 *
 * gmm_uborder_fun.h
 *
 * Code generation for function 'gmm_uborder_fun'
 *
 */

#ifndef GMM_UBORDER_FUN_H
#define GMM_UBORDER_FUN_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern double __anon_fcn(const double mu[2], const double sig[2], const double
  amp[2], double x);
extern void b_normpdfs(double x, const emxArray_real_T *mu, const
  emxArray_real_T *sig, const emxArray_real_T *amp, emxArray_real_T *vals);
extern void normpdfs(double x, const double mu[2], const double sig[2], const
                     double amp[2], double vals[2]);

#endif

/* End of code generation (gmm_uborder_fun.h) */
