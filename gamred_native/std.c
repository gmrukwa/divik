/*
 * File: std.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "std.h"
#include "fetch_thresholds_emxutil.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *x
 * Return Type  : double
 */
double b_std(const emxArray_real_T *x)
{
  double y;
  int n;
  double xbar;
  int k;
  emxArray_real_T *absdiff;
  int kend;
  double absxk;
  double t;
  n = x->size[0];
  if (x->size[0] == 0) {
    y = rtNaN;
  } else if (x->size[0] == 1) {
    if ((!rtIsInf(x->data[0])) && (!rtIsNaN(x->data[0]))) {
      y = 0.0;
    } else {
      y = rtNaN;
    }
  } else {
    xbar = x->data[0];
    for (k = 2; k <= n; k++) {
      xbar += x->data[k - 1];
    }

    emxInit_real_T(&absdiff, 1);
    xbar /= (double)x->size[0];
    kend = absdiff->size[0];
    absdiff->size[0] = x->size[0];
    emxEnsureCapacity_real_T(absdiff, kend);
    for (k = 0; k < n; k++) {
      absdiff->data[k] = fabs(x->data[k] - xbar);
    }

    y = 0.0;
    xbar = 3.3121686421112381E-170;
    kend = x->size[0];
    for (k = 0; k < kend; k++) {
      absxk = absdiff->data[k];
      if (absxk > xbar) {
        t = xbar / absxk;
        y = 1.0 + y * t * t;
        xbar = absxk;
      } else {
        t = absxk / xbar;
        y += t * t;
      }
    }

    emxFree_real_T(&absdiff);
    y = xbar * sqrt(y);
    y /= sqrt((double)x->size[0] - 1.0);
  }

  return y;
}

/*
 * File trailer for std.c
 *
 * [EOF]
 */
