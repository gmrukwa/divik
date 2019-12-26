/*
 * File: g_mix_est_fast_lik.h
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

#ifndef G_MIX_EST_FAST_LIK_H
#define G_MIX_EST_FAST_LIK_H

/* Include Files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void g_mix_est_fast_lik(const emxArray_real_T *raw_sample, double KS,
  const emxArray_real_T *muv, const emxArray_real_T *sigv, const emxArray_real_T
  *pp, emxArray_real_T *ppsort, emxArray_real_T *musort, emxArray_real_T
  *sigsort, double *l_lik);

#endif

/*
 * File trailer for g_mix_est_fast_lik.h
 *
 * [EOF]
 */
