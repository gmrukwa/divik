/*
 * File: dyn_pr_split_w_aux.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "dyn_pr_split_w_aux.h"
#include "fetch_thresholds_emxutil.h"
#include "my_qu_ix_w.h"

/* Function Definitions */

/*
 * function aux_mx=dyn_pr_split_w_aux(data,ygreki)
 * Arguments    : const emxArray_real_T *data
 *                const emxArray_real_T *ygreki
 *                emxArray_real_T *aux_mx
 * Return Type  : void
 */
void dyn_pr_split_w_aux(const emxArray_real_T *data, const emxArray_real_T
  *ygreki, emxArray_real_T *aux_mx)
{
  int i6;
  int loop_ub;
  emxArray_real_T *b_data;
  emxArray_real_T *b_ygreki;
  int kk;
  int i7;
  int jj;
  int b_jj;
  int i8;
  int i9;
  int i10;
  int i11;
  int i12;

  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  */
  /*  */
  /*          auxiliary - compute quality index matrix */
  /*  */
  /* 'dyn_pr_split_w_aux:8' N=length(data); */
  /*  aux_mx */
  /* 'dyn_pr_split_w_aux:11' aux_mx=zeros(N,N); */
  i6 = aux_mx->size[0] * aux_mx->size[1];
  aux_mx->size[0] = data->size[1];
  aux_mx->size[1] = data->size[1];
  emxEnsureCapacity_real_T(aux_mx, i6);
  loop_ub = data->size[1] * data->size[1];
  for (i6 = 0; i6 < loop_ub; i6++) {
    aux_mx->data[i6] = 0.0;
  }

  /* 'dyn_pr_split_w_aux:12' for kk=1:N-1 */
  i6 = data->size[1];
  emxInit_real_T(&b_data, 2);
  emxInit_real_T(&b_ygreki, 2);
  for (kk = 0; kk <= i6 - 2; kk++) {
    /* 'dyn_pr_split_w_aux:13' for jj=kk+1:N */
    i7 = data->size[1] - kk;
    for (jj = 0; jj <= i7 - 2; jj++) {
      b_jj = kk + jj;

      /* 'dyn_pr_split_w_aux:14' aux_mx(kk,jj)= my_qu_ix_w(data(kk:jj-1),ygreki(kk:jj-1)); */
      if (1 + kk > b_jj + 1) {
        i8 = 0;
        i9 = -1;
        i10 = 0;
        i11 = -1;
      } else {
        i8 = kk;
        i9 = b_jj;
        i10 = kk;
        i11 = b_jj;
      }

      i12 = b_data->size[0] * b_data->size[1];
      b_data->size[0] = 1;
      loop_ub = i9 - i8;
      b_data->size[1] = loop_ub + 1;
      emxEnsureCapacity_real_T(b_data, i12);
      for (i9 = 0; i9 <= loop_ub; i9++) {
        b_data->data[i9] = data->data[i8 + i9];
      }

      i8 = b_ygreki->size[0] * b_ygreki->size[1];
      b_ygreki->size[0] = 1;
      loop_ub = i11 - i10;
      b_ygreki->size[1] = loop_ub + 1;
      emxEnsureCapacity_real_T(b_ygreki, i8);
      for (i8 = 0; i8 <= loop_ub; i8++) {
        b_ygreki->data[i8] = ygreki->data[i10 + i8];
      }

      aux_mx->data[kk + aux_mx->size[0] * (b_jj + 1)] = my_qu_ix_w(b_data,
        b_ygreki);
    }
  }

  emxFree_real_T(&b_ygreki);
  emxFree_real_T(&b_data);
}

/*
 * File trailer for dyn_pr_split_w_aux.c
 *
 * [EOF]
 */
