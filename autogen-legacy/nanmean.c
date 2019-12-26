/*
 * File: nanmean.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "nanmean.h"

/* Function Definitions */

/*
 * Arguments    : const emxArray_real_T *varargin_1
 * Return Type  : double
 */
double nanmean(const emxArray_real_T *varargin_1)
{
  double y;
  int vlen;
  int c;
  int k;
  if (varargin_1->size[0] == 0) {
    y = rtNaN;
  } else {
    vlen = varargin_1->size[0];
    y = 0.0;
    c = 0;
    for (k = 0; k < vlen; k++) {
      if (!rtIsNaN(varargin_1->data[k])) {
        y += varargin_1->data[k];
        c++;
      }
    }

    if (c == 0) {
      y = rtNaN;
    } else {
      y /= (double)c;
    }
  }

  return y;
}

/*
 * File trailer for nanmean.c
 *
 * [EOF]
 */
