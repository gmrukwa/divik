/*
 * File: my_qu_ix_w.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "my_qu_ix_w.h"
#include "fetch_thresholds_emxutil.h"
#include "unaryMinOrMax.h"
#include "sqrt.h"
#include "sum.h"
#include "g_mix_est_fast_lik.h"
#include "fetch_thresholds_rtwutil.h"

/* Function Definitions */

/*
 * function wyn=my_qu_ix_w(invec,yinwec)
 * Arguments    : const emxArray_real_T *invec
 *                const emxArray_real_T *yinwec
 * Return Type  : double
 */
double my_qu_ix_w(const emxArray_real_T *invec, const emxArray_real_T *yinwec)
{
  double wyn;
  emxArray_real_T *wwec;
  emxArray_real_T *z1;
  emxArray_real_T *b_invec;
  int nx;
  emxArray_real_T b_yinwec;
  int c_yinwec[1];
  int d_yinwec[1];
  double d0;
  int i13;
  unsigned short a_idx_0;
  int k;
  int n;
  double ex;
  boolean_T exitg1;
  int e_yinwec[1];

  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  */
  /*         quality index */
  /*  */
  /*  */
  /* 'my_qu_ix_w:7' PAR=1; */
  /* 'my_qu_ix_w:8' PAR_sig_min=0.1; */
  /* 'my_qu_ix_w:9' invec=invec(:); */
  /* 'my_qu_ix_w:10' yinwec=yinwec(:); */
  /* 'my_qu_ix_w:11' if (invec(length(invec))-invec(1))<=PAR_sig_min || sum(yinwec)<=1.0e-3 */
  emxInit_real_T(&wwec, 1);
  emxInit_real_T(&z1, 1);
  emxInit_real_T(&b_invec, 1);
  if (invec->data[invec->size[1] - 1] - invec->data[0] <= 0.1) {
    /* 'my_qu_ix_w:12' wyn=inf; */
    wyn = rtInf;
  } else {
    nx = yinwec->size[1];
    b_yinwec = *yinwec;
    c_yinwec[0] = nx;
    b_yinwec.size = &c_yinwec[0];
    b_yinwec.numDimensions = 1;
    if (sum(&b_yinwec) <= 0.001) {
      /* 'my_qu_ix_w:12' wyn=inf; */
      wyn = rtInf;
    } else {
      /* 'my_qu_ix_w:13' else */
      /* 'my_qu_ix_w:14' wwec=yinwec/(sum(yinwec)); */
      nx = yinwec->size[1];
      b_yinwec = *yinwec;
      d_yinwec[0] = nx;
      b_yinwec.size = &d_yinwec[0];
      b_yinwec.numDimensions = 1;
      d0 = sum(&b_yinwec);
      i13 = wwec->size[0];
      wwec->size[0] = yinwec->size[1];
      emxEnsureCapacity_real_T(wwec, i13);
      nx = yinwec->size[1];
      for (i13 = 0; i13 < nx; i13++) {
        wwec->data[i13] = yinwec->data[i13] / d0;
      }

      /* 'my_qu_ix_w:15' wyn1=(PAR+sqrt(sum(((invec-sum(invec.*wwec)).^2).*wwec)))/(max(invec)-min(invec)); */
      nx = invec->size[1];
      i13 = b_invec->size[0];
      b_invec->size[0] = nx;
      emxEnsureCapacity_real_T(b_invec, i13);
      for (i13 = 0; i13 < nx; i13++) {
        b_invec->data[i13] = invec->data[i13] * wwec->data[i13];
      }

      d0 = sum(b_invec);
      i13 = b_invec->size[0];
      b_invec->size[0] = invec->size[1];
      emxEnsureCapacity_real_T(b_invec, i13);
      nx = invec->size[1];
      for (i13 = 0; i13 < nx; i13++) {
        b_invec->data[i13] = invec->data[i13] - d0;
      }

      a_idx_0 = (unsigned short)b_invec->size[0];
      i13 = z1->size[0];
      z1->size[0] = a_idx_0;
      emxEnsureCapacity_real_T(z1, i13);
      a_idx_0 = (unsigned short)b_invec->size[0];
      nx = a_idx_0;
      for (k = 0; k < nx; k++) {
        z1->data[k] = rt_powd_snf(b_invec->data[k], 2.0);
      }

      n = invec->size[1];
      if (invec->size[1] <= 2) {
        if (invec->size[1] == 1) {
          ex = invec->data[0];
        } else if ((invec->data[0] > invec->data[1]) || (rtIsNaN(invec->data[0])
                    && (!rtIsNaN(invec->data[1])))) {
          ex = invec->data[1];
        } else {
          ex = invec->data[0];
        }
      } else {
        if (!rtIsNaN(invec->data[0])) {
          nx = 1;
        } else {
          nx = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= invec->size[1])) {
            if (!rtIsNaN(invec->data[k - 1])) {
              nx = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (nx == 0) {
          ex = invec->data[0];
        } else {
          ex = invec->data[nx - 1];
          i13 = nx + 1;
          for (k = i13; k <= n; k++) {
            if (ex > invec->data[k - 1]) {
              ex = invec->data[k - 1];
            }
          }
        }
      }

      i13 = z1->size[0];
      emxEnsureCapacity_real_T(z1, i13);
      nx = z1->size[0];
      for (i13 = 0; i13 < nx; i13++) {
        z1->data[i13] *= wwec->data[i13];
      }

      d0 = sum(z1);
      b_sqrt(&d0);
      nx = invec->size[1];
      b_yinwec = *invec;
      e_yinwec[0] = nx;
      b_yinwec.size = &e_yinwec[0];
      b_yinwec.numDimensions = 1;
      wyn = (1.0 + d0) / (minOrMaxRealFloatVector(&b_yinwec) - ex);

      /* 'my_qu_ix_w:16' wyn=wyn1; */
    }
  }

  emxFree_real_T(&b_invec);
  emxFree_real_T(&z1);
  emxFree_real_T(&wwec);
  return wyn;
}

/*
 * File trailer for my_qu_ix_w.c
 *
 * [EOF]
 */
