/*
 *
 * fetch_thresholds_terminate.c
 *
 * Code generation for function 'gaussian_mixture_simple_terminate'
 *
 */

/* Include files */
#include "gaussian_mixture_simple_terminate.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void gaussian_mixture_simple_terminate(void)
{
  omp_destroy_nest_lock(&emlrtNestLockGlobal);
  isInitialized_gaussian_mixture_simple = false;
}

/* End of code generation (gaussian_mixture_simple_terminate.c) */
