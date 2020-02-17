/*
 *
 * dyn_pr_split_w.h
 *
 * Code generation for function 'dyn_pr_split_w'
 *
 */

#ifndef DYN_PR_SPLIT_W_H
#define DYN_PR_SPLIT_W_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void dyn_pr_split_w(const emxArray_real_T *data, const emxArray_real_T
  *ygreki, double K_gr, const emxArray_real_T *aux_mx, emxArray_real_T *Q,
  emxArray_real_T *opt_part);

#endif

/* End of code generation (dyn_pr_split_w.h) */
