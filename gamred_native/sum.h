/*
 *
 * sum.h
 *
 * Code generation for function 'sum'
 *
 */

#ifndef SUM_H
#define SUM_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void b_sum(const emxArray_real_T *x, emxArray_real_T *y);
extern void sum(const emxArray_real_T *x, emxArray_real_T *y);

#endif

/* End of code generation (sum.h) */
