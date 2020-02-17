/*
 *
 * fetch_thresholds_initialize.c
 *
 * Code generation for function 'fetch_thresholds_initialize'
 *
 */

/* Include files */
#include "fetch_thresholds_initialize.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void fetch_thresholds_initialize(void)
{
  rt_InitInfAndNaN();
  omp_init_nest_lock(&emlrtNestLockGlobal);
  isInitialized_fetch_thresholds = true;
}

/* End of code generation (fetch_thresholds_initialize.c) */
