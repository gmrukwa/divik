/*
 *
 * sort.h
 *
 * Code generation for function 'sort'
 *
 */

#ifndef SORT_H
#define SORT_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "fetch_thresholds_types.h"

/* Function Declarations */
extern void b_sort(emxArray_real_T *x);
extern void sort(emxArray_real_T *x, emxArray_int32_T *idx);

#endif

/* End of code generation (sort.h) */
