/*
 *
 * gaussian_mixture_simple_initialize.c
 *
 * Code generation for function 'gaussian_mixture_simple_initialize'
 *
 */

/* Include files */
#include "gaussian_mixture_simple_initialize.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void gaussian_mixture_simple_initialize(void)
{
  rt_InitInfAndNaN();
  omp_init_nest_lock(&emlrtNestLockGlobal);
  isInitialized_gaussian_mixture_simple = true;
}

/* End of code generation (gaussian_mixture_simple_initialize.c) */
