/*
 *
 * histcounts.c
 *
 * Code generation for function 'histcounts'
 *
 */

/* Include files */
#include "histcounts.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "fetch_thresholds_rtwutil.h"
#include "g_mix_est_fast_lik.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */

/*
 *
 */
void histcounts(const emxArray_real_T *x, double varargin_1, emxArray_real_T *n,
                emxArray_real_T *edges)
{
  int nx;
  int k;
  double LowLimit;
  int nbinsActual;
  double HighLimit;
  int i;
  int high_i;
  int low_ip1;
  double xScale;
  bool b;
  bool b1;
  double epsxScale;
  int mid_i;
  double leftEdge;
  double RawBinWidth;
  emxArray_int32_T *ni;
  int exponent;
  bool guard1 = false;
  nx = x->size[0];
  k = 0;
  while ((k + 1 <= nx) && (rtIsInf(x->data[k]) || rtIsNaN(x->data[k]))) {
    k++;
  }

  if (k + 1 > x->size[0]) {
    LowLimit = 0.0;
    nbinsActual = 0;
  } else {
    LowLimit = x->data[k];
    nbinsActual = 1;
  }

  HighLimit = LowLimit;
  i = k + 2;
  for (high_i = i; high_i <= nx; high_i++) {
    xScale = x->data[high_i - 1];
    if ((!rtIsInf(xScale)) && (!rtIsNaN(xScale))) {
      if (xScale < LowLimit) {
        LowLimit = xScale;
      } else {
        if (xScale > HighLimit) {
          HighLimit = xScale;
        }
      }

      nbinsActual++;
    }
  }

  low_ip1 = (int)varargin_1;
  if (nbinsActual > 0) {
    xScale = fmax(fabs(LowLimit), fabs(HighLimit));
    b = !rtIsInf(xScale);
    b1 = !rtIsNaN(xScale);
    if (b && b1) {
      if (xScale <= 2.2250738585072014E-308) {
        epsxScale = 4.94065645841247E-324;
      } else {
        frexp(xScale, &mid_i);
        epsxScale = ldexp(1.0, mid_i - 53);
      }
    } else {
      epsxScale = rtNaN;
    }

    leftEdge = HighLimit - LowLimit;
    RawBinWidth = fmax(leftEdge / (double)low_ip1, epsxScale);
    if (leftEdge > fmax(sqrt(epsxScale), 2.2250738585072014E-308)) {
      xScale = rt_powd_snf(10.0, floor(log10(RawBinWidth)));
      epsxScale = xScale * floor(RawBinWidth / xScale);
      leftEdge = fmax(fmin(epsxScale * floor(LowLimit / epsxScale), LowLimit),
                      -1.7976931348623157E+308);
      nbinsActual = (int)varargin_1 + 1;
      if (low_ip1 > 1) {
        epsxScale = (int)varargin_1;
        xScale = (HighLimit - leftEdge) / epsxScale;
        epsxScale = rt_powd_snf(10.0, floor(log10((HighLimit - leftEdge) /
          (epsxScale - 1.0) - xScale)));
        epsxScale *= ceil(xScale / epsxScale);
      }

      xScale = fmin(fmax(leftEdge + (double)(int)varargin_1 * epsxScale,
                         HighLimit), 1.7976931348623157E+308);
    } else {
      nbinsActual = (int)varargin_1 + 1;
      if (b && b1) {
        if (xScale <= 2.2250738585072014E-308) {
          xScale = 4.94065645841247E-324;
        } else {
          frexp(xScale, &exponent);
          xScale = ldexp(1.0, exponent - 53);
        }
      } else {
        xScale = rtNaN;
      }

      epsxScale = (int)varargin_1;
      xScale = fmax(1.0, ceil(epsxScale * xScale));
      leftEdge = floor(2.0 * (LowLimit - xScale / 4.0)) / 2.0;
      xScale = ceil(2.0 * (HighLimit + xScale / 4.0)) / 2.0;
      epsxScale = (xScale - leftEdge) / epsxScale;
    }

    if ((!rtIsInf(epsxScale)) && (!rtIsNaN(epsxScale))) {
      i = edges->size[0] * edges->size[1];
      edges->size[0] = 1;
      edges->size[1] = nbinsActual;
      emxEnsureCapacity_real_T(edges, i);
      for (i = 0; i < nbinsActual; i++) {
        edges->data[i] = 0.0;
      }

      edges->data[0] = leftEdge;
      for (high_i = 0; high_i <= nbinsActual - 3; high_i++) {
        edges->data[high_i + 1] = leftEdge + ((double)high_i + 1.0) * epsxScale;
      }

      edges->data[edges->size[1] - 1] = xScale;
    } else {
      i = edges->size[0] * edges->size[1];
      edges->size[0] = 1;
      edges->size[1] = nbinsActual;
      emxEnsureCapacity_real_T(edges, i);
      edges->data[nbinsActual - 1] = xScale;
      if (edges->size[1] >= 2) {
        edges->data[0] = leftEdge;
        if (edges->size[1] >= 3) {
          if ((leftEdge == -xScale) && (nbinsActual > 2)) {
            i = nbinsActual - 1;
            for (k = 2; k <= i; k++) {
              edges->data[k - 1] = xScale * (((double)((k << 1) - nbinsActual) -
                1.0) / ((double)nbinsActual - 1.0));
            }

            if ((nbinsActual & 1) == 1) {
              edges->data[nbinsActual >> 1] = 0.0;
            }
          } else if (((leftEdge < 0.0) != (xScale < 0.0)) && ((fabs(leftEdge) >
                       8.9884656743115785E+307) || (fabs(xScale) >
                       8.9884656743115785E+307))) {
            epsxScale = leftEdge / ((double)edges->size[1] - 1.0);
            xScale /= (double)edges->size[1] - 1.0;
            i = edges->size[1];
            for (k = 0; k <= i - 3; k++) {
              edges->data[k + 1] = (leftEdge + xScale * ((double)k + 1.0)) -
                epsxScale * ((double)k + 1.0);
            }
          } else {
            epsxScale = (xScale - leftEdge) / ((double)edges->size[1] - 1.0);
            i = edges->size[1];
            for (k = 0; k <= i - 3; k++) {
              edges->data[k + 1] = leftEdge + ((double)k + 1.0) * epsxScale;
            }
          }
        }
      }
    }
  } else if (low_ip1 >= 2) {
    i = edges->size[0] * edges->size[1];
    edges->size[0] = 1;
    high_i = low_ip1 + 1;
    edges->size[1] = high_i;
    emxEnsureCapacity_real_T(edges, i);
    for (i = 0; i < high_i; i++) {
      edges->data[i] = 0.0;
    }

    for (k = 0; k <= low_ip1; k++) {
      edges->data[k] = k;
    }
  } else {
    i = edges->size[0] * edges->size[1];
    edges->size[0] = 1;
    edges->size[1] = 2;
    emxEnsureCapacity_real_T(edges, i);
    edges->data[0] = 0.0;
    edges->data[1] = 1.0;
  }

  emxInit_int32_T(&ni, 2);
  i = ni->size[0] * ni->size[1];
  ni->size[0] = 1;
  ni->size[1] = edges->size[1] - 1;
  emxEnsureCapacity_int32_T(ni, i);
  high_i = edges->size[1] - 1;
  for (i = 0; i < high_i; i++) {
    ni->data[i] = 0;
  }

  nx = x->size[0];
  leftEdge = edges->data[0];
  xScale = edges->data[1] - edges->data[0];
  for (k = 0; k < nx; k++) {
    if ((x->data[k] >= leftEdge) && (x->data[k] <= edges->data[edges->size[1] -
         1])) {
      epsxScale = ceil((x->data[k] - leftEdge) / xScale);
      guard1 = false;
      if ((epsxScale >= 1.0) && (epsxScale < edges->size[1])) {
        i = (int)epsxScale - 1;
        if ((x->data[k] >= edges->data[i]) && (x->data[k] < edges->data[(int)
             epsxScale])) {
          ni->data[i]++;
        } else {
          guard1 = true;
        }
      } else {
        guard1 = true;
      }

      if (guard1) {
        high_i = edges->size[1];
        nbinsActual = 1;
        low_ip1 = 2;
        while (high_i > low_ip1) {
          mid_i = (nbinsActual >> 1) + (high_i >> 1);
          if (((nbinsActual & 1) == 1) && ((high_i & 1) == 1)) {
            mid_i++;
          }

          if (x->data[k] >= edges->data[mid_i - 1]) {
            nbinsActual = mid_i;
            low_ip1 = mid_i + 1;
          } else {
            high_i = mid_i;
          }
        }

        ni->data[nbinsActual - 1]++;
      }
    }
  }

  i = n->size[0] * n->size[1];
  n->size[0] = 1;
  n->size[1] = ni->size[1];
  emxEnsureCapacity_real_T(n, i);
  high_i = ni->size[0] * ni->size[1];
  for (i = 0; i < high_i; i++) {
    n->data[i] = ni->data[i];
  }

  emxFree_int32_T(&ni);
}

/* End of code generation (histcounts.c) */
