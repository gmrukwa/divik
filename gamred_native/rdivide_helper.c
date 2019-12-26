/*
 * File: rdivide_helper.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "rdivide_helper.h"
#include "fetch_thresholds_emxutil.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *x
 *                const emxArray_real_T *y
 *                emxArray_real_T *z
 * Return Type  : void
 */
void rdivide_helper(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *z)
{
  int i15;
  int loop_ub;
  i15 = z->size[0];
  z->size[0] = x->size[0];
  emxEnsureCapacity_real_T(z, i15);
  loop_ub = x->size[0];
  for (i15 = 0; i15 < loop_ub; i15++) {
    z->data[i15] = x->data[i15] / y->data[i15];
  }
}

/*
 * File trailer for rdivide_helper.c
 *
 * [EOF]
 */
