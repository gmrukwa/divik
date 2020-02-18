/*
 *
 * my_qu_ix_w.c
 *
 * Code generation for function 'my_qu_ix_w'
 *
 */

/* Include files */
#include "my_qu_ix_w.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "fetch_thresholds_rtwutil.h"
#include "g_mix_est_fast_lik.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */

/*
 * function wyn=my_qu_ix_w(invec,yinwec)
 */
double my_qu_ix_w(const emxArray_real_T *invec, const emxArray_real_T *yinwec)
{
  double wyn;
  emxArray_real_T *wwec;
  emxArray_real_T *x;
  emxArray_real_T *y;
  emxArray_real_T *a;
  int vlen;
  double b_y;
  int k;
  int i;
  int n;
  double ex;
  bool exitg1;
  double b_ex;

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
  emxInit_real_T(&x, 1);
  emxInit_real_T(&y, 1);
  emxInit_real_T(&a, 1);
  if (invec->data[invec->size[1] - 1] - invec->data[0] <= 0.1) {
    /* 'my_qu_ix_w:12' wyn=inf; */
    wyn = rtInf;
  } else {
    vlen = yinwec->size[1];
    if (yinwec->size[1] == 0) {
      b_y = 0.0;
    } else {
      b_y = yinwec->data[0];
      for (k = 2; k <= vlen; k++) {
        b_y += yinwec->data[k - 1];
      }
    }

    if (b_y <= 0.001) {
      /* 'my_qu_ix_w:12' wyn=inf; */
      wyn = rtInf;
    } else {
      /* 'my_qu_ix_w:13' else */
      /* 'my_qu_ix_w:14' wwec=yinwec/(sum(yinwec)); */
      i = wwec->size[0];
      wwec->size[0] = yinwec->size[1];
      emxEnsureCapacity_real_T(wwec, i);
      vlen = yinwec->size[1];
      for (i = 0; i < vlen; i++) {
        wwec->data[i] = yinwec->data[i] / b_y;
      }

      /* 'my_qu_ix_w:15' wyn1=(PAR+sqrt(sum(((invec-sum(invec.*wwec)).^2).*wwec)))/(max(invec)-min(invec)); */
      n = invec->size[1];
      if (invec->size[1] <= 2) {
        if (invec->size[1] == 1) {
          ex = invec->data[0];
        } else if ((invec->data[0] < invec->data[1]) || (rtIsNaN(invec->data[0])
                    && (!rtIsNaN(invec->data[1])))) {
          ex = invec->data[1];
        } else {
          ex = invec->data[0];
        }
      } else {
        if (!rtIsNaN(invec->data[0])) {
          vlen = 1;
        } else {
          vlen = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= invec->size[1])) {
            if (!rtIsNaN(invec->data[k - 1])) {
              vlen = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (vlen == 0) {
          ex = invec->data[0];
        } else {
          ex = invec->data[vlen - 1];
          i = vlen + 1;
          for (k = i; k <= n; k++) {
            if (ex < invec->data[k - 1]) {
              ex = invec->data[k - 1];
            }
          }
        }
      }

      n = invec->size[1];
      if (invec->size[1] <= 2) {
        if (invec->size[1] == 1) {
          b_ex = invec->data[0];
        } else if ((invec->data[0] > invec->data[1]) || (rtIsNaN(invec->data[0])
                    && (!rtIsNaN(invec->data[1])))) {
          b_ex = invec->data[1];
        } else {
          b_ex = invec->data[0];
        }
      } else {
        if (!rtIsNaN(invec->data[0])) {
          vlen = 1;
        } else {
          vlen = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= invec->size[1])) {
            if (!rtIsNaN(invec->data[k - 1])) {
              vlen = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (vlen == 0) {
          b_ex = invec->data[0];
        } else {
          b_ex = invec->data[vlen - 1];
          i = vlen + 1;
          for (k = i; k <= n; k++) {
            b_y = invec->data[k - 1];
            if (b_ex > b_y) {
              b_ex = b_y;
            }
          }
        }
      }

      i = x->size[0];
      x->size[0] = invec->size[1];
      emxEnsureCapacity_real_T(x, i);
      vlen = invec->size[1];
      for (i = 0; i < vlen; i++) {
        x->data[i] = invec->data[i] * wwec->data[i];
      }

      vlen = x->size[0];
      if (x->size[0] == 0) {
        b_y = 0.0;
      } else {
        b_y = x->data[0];
        for (k = 2; k <= vlen; k++) {
          b_y += x->data[k - 1];
        }
      }

      i = a->size[0];
      a->size[0] = invec->size[1];
      emxEnsureCapacity_real_T(a, i);
      vlen = invec->size[1];
      for (i = 0; i < vlen; i++) {
        a->data[i] = invec->data[i] - b_y;
      }

      i = y->size[0];
      y->size[0] = a->size[0];
      emxEnsureCapacity_real_T(y, i);
      vlen = a->size[0];
      for (k = 0; k < vlen; k++) {
        y->data[k] = rt_powd_snf(a->data[k], 2.0);
      }

      i = x->size[0];
      x->size[0] = y->size[0];
      emxEnsureCapacity_real_T(x, i);
      vlen = y->size[0];
      for (i = 0; i < vlen; i++) {
        x->data[i] = y->data[i] * wwec->data[i];
      }

      vlen = x->size[0];
      if (x->size[0] == 0) {
        b_y = 0.0;
      } else {
        b_y = x->data[0];
        for (k = 2; k <= vlen; k++) {
          b_y += x->data[k - 1];
        }
      }

      wyn = (sqrt(b_y) + 1.0) / (ex - b_ex);

      /* 'my_qu_ix_w:16' wyn=wyn1; */
    }
  }

  emxFree_real_T(&a);
  emxFree_real_T(&y);
  emxFree_real_T(&x);
  emxFree_real_T(&wwec);
  return wyn;
}

/* End of code generation (my_qu_ix_w.c) */
