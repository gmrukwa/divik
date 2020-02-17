/*
 *
 * g_mix_est_fast_lik.h
 *
 * Code generation for function 'g_mix_est_fast_lik'
 *
 */

#ifndef G_MIX_EST_FAST_LIK_H
#define G_MIX_EST_FAST_LIK_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void g_mix_est_fast_lik(const emxArray_real_T *raw_sample, double KS,
  const emxArray_real_T *muv, const emxArray_real_T *sigv, const emxArray_real_T
  *pp, emxArray_real_T *ppsort, emxArray_real_T *musort, emxArray_real_T
  *sigsort, double *l_lik);

#endif

/* End of code generation (g_mix_est_fast_lik.h) */
