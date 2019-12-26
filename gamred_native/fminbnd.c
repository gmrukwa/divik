/*
 * File: fminbnd.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "fminbnd.h"
#include "gmm_uborder_fun.h"

/* Function Definitions */

/*
 * Arguments    : coder_internal_ref * const funfcnInput_tunableEnvironment[3]
 *                double ax
 *                double bx
 * Return Type  : double
 */
double fminbnd(coder_internal_ref * const funfcnInput_tunableEnvironment[3],
               double ax, double bx)
{
  double xf;
  int iter;
  double a;
  double b;
  double v;
  double w;
  double d;
  double e;
  double fx;
  int funccount;
  double fv;
  double fw;
  double xm;
  double tol1;
  double tol2;
  boolean_T exitg1;
  int gs;
  double p;
  double r;
  double x;
  double q;
  double b_p;
  iter = 0;
  if (ax > bx) {
    xf = rtNaN;
    fun(funfcnInput_tunableEnvironment[0], funfcnInput_tunableEnvironment[1],
        funfcnInput_tunableEnvironment[2], rtNaN);
  } else {
    a = ax;
    b = bx;
    v = ax + 0.3819660112501051 * (bx - ax);
    w = v;
    xf = v;
    d = 0.0;
    e = 0.0;
    fx = fun(funfcnInput_tunableEnvironment[0], funfcnInput_tunableEnvironment[1],
             funfcnInput_tunableEnvironment[2], v);
    funccount = 1;
    fv = fx;
    fw = fx;
    xm = 0.5 * (ax + bx);
    tol1 = 1.4901161193847656E-8 * fabs(v) + 3.3333333333333335E-5;
    tol2 = 2.0 * tol1;
    exitg1 = false;
    while ((!exitg1) && (fabs(xf - xm) > tol2 - 0.5 * (b - a))) {
      gs = 1;
      if (fabs(e) > tol1) {
        gs = 0;
        p = xf - w;
        r = p * (fx - fv);
        x = xf - v;
        q = x * (fx - fw);
        p = x * q - p * r;
        q = 2.0 * (q - r);
        if (q > 0.0) {
          p = -p;
        }

        q = fabs(q);
        r = e;
        e = d;
        if ((fabs(p) < fabs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b -
              xf))) {
          d = p / q;
          x = xf + d;
          if ((x - a < tol2) || (b - x < tol2)) {
            p = xm - xf;
            x = p;
            if (p < 0.0) {
              x = -1.0;
            } else if (p > 0.0) {
              x = 1.0;
            } else {
              if (p == 0.0) {
                x = 0.0;
              }
            }

            d = tol1 * (x + (double)(p == 0.0));
          }
        } else {
          gs = 1;
        }
      }

      if (gs != 0) {
        if (xf >= xm) {
          e = a - xf;
        } else {
          e = b - xf;
        }

        d = 0.3819660112501051 * e;
      }

      x = d;
      if (d < 0.0) {
        x = -1.0;
      } else if (d > 0.0) {
        x = 1.0;
      } else {
        if (d == 0.0) {
          x = 0.0;
        }
      }

      p = fabs(d);
      if ((p > tol1) || rtIsNaN(tol1)) {
        b_p = p;
      } else {
        b_p = tol1;
      }

      x = xf + (x + (double)(d == 0.0)) * b_p;
      p = fun(funfcnInput_tunableEnvironment[0], funfcnInput_tunableEnvironment
              [1], funfcnInput_tunableEnvironment[2], x);
      funccount++;
      iter++;
      if (p <= fx) {
        if (x >= xf) {
          a = xf;
        } else {
          b = xf;
        }

        v = w;
        fv = fw;
        w = xf;
        fw = fx;
        xf = x;
        fx = p;
      } else {
        if (x < xf) {
          a = x;
        } else {
          b = x;
        }

        if ((p <= fw) || (w == xf)) {
          v = w;
          fv = fw;
          w = x;
          fw = p;
        } else {
          if ((p <= fv) || (v == xf) || (v == w)) {
            v = x;
            fv = p;
          }
        }
      }

      xm = 0.5 * (a + b);
      tol1 = 1.4901161193847656E-8 * fabs(xf) + 3.3333333333333335E-5;
      tol2 = 2.0 * tol1;
      if ((funccount >= 500) || (iter >= 500)) {
        exitg1 = true;
      }
    }
  }

  return xf;
}

/*
 * File trailer for fminbnd.c
 *
 * [EOF]
 */
