/*
 * File: eps.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "eps.h"

/* Function Definitions */

/*
 * Arguments    : double x
 * Return Type  : double
 */
double eps(double x)
{
  double r;
  double absxk;
  int exponent;
  absxk = fabs(x);
  if ((!rtIsInf(absxk)) && (!rtIsNaN(absxk))) {
    if (absxk <= 2.2250738585072014E-308) {
      r = 4.94065645841247E-324;
    } else {
      frexp(absxk, &exponent);
      r = ldexp(1.0, exponent - 53);
    }
  } else {
    r = rtNaN;
  }

  return r;
}

/*
 * File trailer for eps.c
 *
 * [EOF]
 */
