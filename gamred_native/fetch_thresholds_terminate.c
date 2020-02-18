/*
 *
 * fetch_thresholds_terminate.c
 *
 * Code generation for function 'fetch_thresholds_terminate'
 *
 */

/* Include files */
#include "fetch_thresholds_terminate.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void fetch_thresholds_terminate(void)
{
  omp_destroy_nest_lock(&emlrtNestLockGlobal);
  isInitialized_fetch_thresholds = false;
}

/* End of code generation (fetch_thresholds_terminate.c) */
