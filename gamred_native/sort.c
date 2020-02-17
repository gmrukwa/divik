/*
 *
 * sort.c
 *
 * Code generation for function 'sort'
 *
 */

/* Include files */
#include "sort.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "rt_nonfinite.h"
#include "sortIdx.h"

/* Function Definitions */

/*
 *
 */
void b_sort(emxArray_real_T *x)
{
  int dim;
  emxArray_real_T *vwork;
  int j;
  int vlen;
  int vstride;
  int k;
  emxArray_int32_T *b_vwork;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }

  emxInit_real_T(&vwork, 1);
  if (dim + 2 <= 1) {
    j = x->size[0];
  } else {
    j = 1;
  }

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
 *
 */
void sort(emxArray_real_T *x, emxArray_int32_T *idx)
{
  int dim;
  emxArray_real_T *vwork;
  int i;
  int vlen;
  int j;
  int vstride;
  int k;
  emxArray_int32_T *iidx;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }

  emxInit_real_T(&vwork, 1);
  if (dim + 2 <= 1) {
    i = x->size[0];
  } else {
    i = 1;
  }

  vlen = i - 1;
  j = vwork->size[0];
  vwork->size[0] = i;
  emxEnsureCapacity_real_T(vwork, j);
  i = idx->size[0];
  idx->size[0] = x->size[0];
  emxEnsureCapacity_int32_T(idx, i);
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x->size[0];
  }

  emxInit_int32_T(&iidx, 1);
  for (j = 0; j < vstride; j++) {
    for (k = 0; k <= vlen; k++) {
      vwork->data[k] = x->data[j + k * vstride];
    }

    sortIdx(vwork, iidx);
    for (k = 0; k <= vlen; k++) {
      i = j + k * vstride;
      x->data[i] = vwork->data[k];
      idx->data[i] = iidx->data[k];
    }
  }

  emxFree_int32_T(&iidx);
  emxFree_real_T(&vwork);
}

/* End of code generation (sort.c) */
