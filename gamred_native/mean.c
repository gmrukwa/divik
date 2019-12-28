/*
 * File: mean.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "mean.h"
#include "fetch_thresholds_emxutil.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *x
 *                emxArray_real_T *y
 * Return Type  : void
 */
void mean(const emxArray_real_T *x, emxArray_real_T *y)
{
  int npages;
  int xpageoffset;
  int i;
  if (x->size[1] == 0) {
    y->size[0] = 1;
    y->size[1] = 0;
  } else {
    npages = x->size[1];
    xpageoffset = y->size[0] * y->size[1];
    y->size[0] = 1;
    y->size[1] = (unsigned short)x->size[1];
    emxEnsureCapacity_real_T(y, xpageoffset);
    for (i = 0; i < npages; i++) {
      xpageoffset = i << 1;
      y->data[i] = x->data[xpageoffset];
      y->data[i] += x->data[xpageoffset + 1];
    }
  }

  xpageoffset = y->size[0] * y->size[1];
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  emxEnsureCapacity_real_T(y, i);
  i = xpageoffset - 1;
  for (xpageoffset = 0; xpageoffset <= i; xpageoffset++) {
    y->data[xpageoffset] /= 2.0;
  }
}

/*
 * File trailer for mean.c
 *
 * [EOF]
 */
