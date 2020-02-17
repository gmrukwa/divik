/*
 *
 * mean.c
 *
 * Code generation for function 'mean'
 *
 */

/* Include files */
#include "mean.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 *
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
      y->data[i] = x->data[xpageoffset] + x->data[xpageoffset + 1];
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

/* End of code generation (mean.c) */
