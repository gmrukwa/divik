/*
 *
 * sum.c
 *
 * Code generation for function 'sum'
 *
 */

/* Include files */
#include "sum.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 *
 */
void b_sum(const emxArray_real_T *x, emxArray_real_T *y)
{
  int vlen;
  int vstride;
  unsigned int sz_idx_0;
  int k;
  int j;
  int xoffset;
  vlen = x->size[1];
  if (x->size[1] == 0) {
    sz_idx_0 = (unsigned int)x->size[0];
    k = y->size[0];
    y->size[0] = (int)sz_idx_0;
    emxEnsureCapacity_real_T(y, k);
    j = (int)sz_idx_0;
    for (k = 0; k < j; k++) {
      y->data[k] = 0.0;
    }
  } else {
    vstride = x->size[0];
    k = y->size[0];
    y->size[0] = x->size[0];
    emxEnsureCapacity_real_T(y, k);
    for (j = 0; j < vstride; j++) {
      y->data[j] = x->data[j];
    }

    for (k = 2; k <= vlen; k++) {
      xoffset = (k - 1) * vstride;
      for (j = 0; j < vstride; j++) {
        y->data[j] += x->data[xoffset + j];
      }
    }
  }
}

/*
 *
 */
void sum(const emxArray_real_T *x, emxArray_real_T *y)
{
  int vlen;
  int npages;
  int i;
  int xpageoffset;
  int k;
  vlen = x->size[0];
  if (x->size[1] == 0) {
    y->size[0] = 1;
    y->size[1] = 0;
  } else {
    npages = x->size[1];
    i = y->size[0] * y->size[1];
    y->size[0] = 1;
    y->size[1] = x->size[1];
    emxEnsureCapacity_real_T(y, i);
    for (i = 0; i < npages; i++) {
      xpageoffset = i * x->size[0];
      y->data[i] = x->data[xpageoffset];
      for (k = 2; k <= vlen; k++) {
        y->data[i] += x->data[(xpageoffset + k) - 1];
      }
    }
  }
}

/* End of code generation (sum.c) */
