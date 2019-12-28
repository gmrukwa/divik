/*
 * File: abs.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "abs.h"
#include "fetch_thresholds_emxutil.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *x
 *                emxArray_real_T *y
 * Return Type  : void
 */
void b_abs(const emxArray_real_T *x, emxArray_real_T *y)
{
  int nx;
  unsigned int x_idx_0;
  int k;
  nx = x->size[0];
  x_idx_0 = (unsigned int)x->size[0];
  k = y->size[0];
  y->size[0] = (int)x_idx_0;
  emxEnsureCapacity_real_T(y, k);
  for (k = 0; k < nx; k++) {
    y->data[k] = fabs(x->data[k]);
  }
}

/*
 * File trailer for abs.c
 *
 * [EOF]
 */
