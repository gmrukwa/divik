/*
 * File: gaussian_mixture_simple.h
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

#ifndef GAUSSIAN_MIXTURE_SIMPLE_H
#define GAUSSIAN_MIXTURE_SIMPLE_H

/* Include Files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void gaussian_mixture_simple(const emxArray_real_T *x, const
  emxArray_real_T *counts, double KS, emxArray_real_T *pp_est, emxArray_real_T
  *mu_est, emxArray_real_T *sig_est, double *TIC, double *l_lik);

#endif

/*
 * File trailer for gaussian_mixture_simple.h
 *
 * [EOF]
 */
