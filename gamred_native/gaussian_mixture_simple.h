/*
 *
 * gaussian_mixture_simple.h
 *
 * Code generation for function 'gaussian_mixture_simple'
 *
 */

#ifndef GAUSSIAN_MIXTURE_SIMPLE_H
#define GAUSSIAN_MIXTURE_SIMPLE_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void gaussian_mixture_simple(const emxArray_real_T *x, const
  emxArray_real_T *counts, double KS, emxArray_real_T *pp_est, emxArray_real_T
  *mu_est, emxArray_real_T *sig_est, double *TIC, double *l_lik);

#endif

/* End of code generation (gaussian_mixture_simple.h) */
