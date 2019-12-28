/*
 * File: sort1.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "sort1.h"
#include "fetch_thresholds_emxutil.h"
#include "sortIdx.h"

/* Function Definitions */

/*
 * Arguments    : emxArray_real_T *x
 * Return Type  : void
 */
void b_sort(emxArray_real_T *x)
{
  int dim;
  int j;
  emxArray_real_T *vwork;
  int vlen;
  int vstride;
  int k;
  emxArray_int32_T *b_vwork;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }

  if (dim + 2 <= 1) {
    j = x->size[0];
  } else {
    j = 1;
  }

  emxInit_real_T(&vwork, 1);
  vlen = j - 1;
  vstride = vwork->size[0];
  vwork->size[0] = j;
  emxEnsureCapacity_real_T(vwork, vstride);
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x->size[0];
  }

  emxInit_int32_T(&b_vwork, 1);
  for (j = 0; j < vstride; j++) {
    for (k = 0; k <= vlen; k++) {
      vwork->data[k] = x->data[j + k * vstride];
    }

    sortIdx(vwork, b_vwork);
    for (k = 0; k <= vlen; k++) {
      x->data[j + k * vstride] = vwork->data[k];
    }
  }

  emxFree_int32_T(&b_vwork);
  emxFree_real_T(&vwork);
}

/*
 * Arguments    : emxArray_real_T *x
 *                emxArray_int32_T *idx
 * Return Type  : void
 */
void sort(emxArray_real_T *x, emxArray_int32_T *idx)
{
  int dim;
  int i16;
  emxArray_real_T *vwork;
  int vlen;
  int x_idx_0;
  int vstride;
  int k;
  emxArray_int32_T *iidx;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }

  if (dim + 2 <= 1) {
    i16 = x->size[0];
  } else {
    i16 = 1;
  }

  emxInit_real_T(&vwork, 1);
  vlen = i16 - 1;
  x_idx_0 = vwork->size[0];
  vwork->size[0] = i16;
  emxEnsureCapacity_real_T(vwork, x_idx_0);
  x_idx_0 = x->size[0];
  i16 = idx->size[0];
  idx->size[0] = x_idx_0;
  emxEnsureCapacity_int32_T(idx, i16);
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x->size[0];
  }

  emxInit_int32_T(&iidx, 1);
  for (x_idx_0 = 0; x_idx_0 < vstride; x_idx_0++) {
    for (k = 0; k <= vlen; k++) {
      vwork->data[k] = x->data[x_idx_0 + k * vstride];
    }

    sortIdx(vwork, iidx);
    for (k = 0; k <= vlen; k++) {
      i16 = x_idx_0 + k * vstride;
      x->data[i16] = vwork->data[k];
      idx->data[i16] = iidx->data[k];
    }
  }

  emxFree_int32_T(&iidx);
  emxFree_real_T(&vwork);
}

/*
 * File trailer for sort1.c
 *
 * [EOF]
 */
