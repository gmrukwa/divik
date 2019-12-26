/*
 * File: unaryMinOrMax.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "unaryMinOrMax.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *x
 * Return Type  : double
 */
double minOrMaxRealFloatVector(const emxArray_real_T *x)
{
  double ex;
  int n;
  int idx;
  int k;
  boolean_T exitg1;
  n = x->size[0];
  if (x->size[0] <= 2) {
    if (x->size[0] == 1) {
      ex = x->data[0];
    } else if ((x->data[0] < x->data[1]) || (rtIsNaN(x->data[0]) && (!rtIsNaN
                 (x->data[1])))) {
      ex = x->data[1];
    } else {
      ex = x->data[0];
    }
  } else {
    if (!rtIsNaN(x->data[0])) {
      idx = 1;
    } else {
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= x->size[0])) {
        if (!rtIsNaN(x->data[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }

    if (idx == 0) {
      ex = x->data[0];
    } else {
      ex = x->data[idx - 1];
      idx++;
      for (k = idx; k <= n; k++) {
        if (ex < x->data[k - 1]) {
          ex = x->data[k - 1];
        }
      }
    }
  }

  return ex;
}

/*
 * File trailer for unaryMinOrMax.c
 *
 * [EOF]
 */
