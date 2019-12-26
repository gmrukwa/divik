/*
 * File: histcounts.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "histcounts.h"
#include "fetch_thresholds_emxutil.h"
#include "g_mix_est_fast_lik.h"
#include "eps.h"
#include "fetch_thresholds_rtwutil.h"

/* Function Declarations */
static void parseinput(double varargin_1, double BinEdges[2], int *NumBins,
  boolean_T *NumBinsSupplied);

/* Function Definitions */

/*
 * Arguments    : double varargin_1
 *                double BinEdges[2]
 *                int *NumBins
 *                boolean_T *NumBinsSupplied
 * Return Type  : void
 */
static void parseinput(double varargin_1, double BinEdges[2], int *NumBins,
  boolean_T *NumBinsSupplied)
{
  *NumBins = (int)varargin_1;
  *NumBinsSupplied = true;
  BinEdges[0] = 0.0;
  BinEdges[1] = 0.0;
}

/*
 * Arguments    : const emxArray_real_T *x
 *                double varargin_1
 *                emxArray_real_T *n
 *                emxArray_real_T *edges
 * Return Type  : void
 */
void histcounts(const emxArray_real_T *x, double varargin_1, emxArray_real_T *n,
                emxArray_real_T *edges)
{
  double BinEdges[2];
  int nbinsActual;
  boolean_T NumBinsSupplied;
  int nx;
  int k;
  double xLowLimit;
  int low_i;
  double xHighLimit;
  int low_ip1;
  int high_i;
  double binWidth;
  double leftEdge;
  double u0;
  double epsxScale;
  double xScale;
  emxArray_int32_T *ni;
  boolean_T guard1 = false;
  parseinput(varargin_1, BinEdges, &nbinsActual, &NumBinsSupplied);
  nx = x->size[0];
  k = 0;
  while ((k + 1 <= nx) && (rtIsInf(x->data[k]) || rtIsNaN(x->data[k]))) {
    k++;
  }

  if (k + 1 > x->size[0]) {
    xLowLimit = 0.0;
    low_i = 0;
  } else {
    xLowLimit = x->data[k];
    low_i = 1;
  }

  xHighLimit = xLowLimit;
  low_ip1 = k + 2;
  for (high_i = low_ip1; high_i <= nx; high_i++) {
    if ((!rtIsInf(x->data[high_i - 1])) && (!rtIsNaN(x->data[high_i - 1]))) {
      if (x->data[high_i - 1] < xLowLimit) {
        xLowLimit = x->data[high_i - 1];
      } else {
        if (x->data[high_i - 1] > xHighLimit) {
          xHighLimit = x->data[high_i - 1];
        }
      }

      low_i++;
    }
  }

  binWidth = xHighLimit - xLowLimit;
  leftEdge = binWidth / (double)nbinsActual;
  if (low_i > 0) {
    u0 = fabs(xLowLimit);
    epsxScale = fabs(xHighLimit);
    if ((u0 > epsxScale) || rtIsNaN(epsxScale)) {
      xScale = u0;
    } else {
      xScale = epsxScale;
    }

    epsxScale = eps(xScale);
    if ((!(leftEdge > epsxScale)) && (!rtIsNaN(epsxScale))) {
      leftEdge = epsxScale;
    }

    u0 = sqrt(epsxScale);
    if (!(u0 > 2.2250738585072014E-308)) {
      u0 = 2.2250738585072014E-308;
    }

    if (binWidth > u0) {
      epsxScale = rt_powd_snf(10.0, floor(log10(leftEdge)));
      binWidth = epsxScale * floor(leftEdge / epsxScale);
      u0 = binWidth * floor(xLowLimit / binWidth);
      if ((u0 < xLowLimit) || rtIsNaN(xLowLimit)) {
        leftEdge = u0;
      } else {
        leftEdge = xLowLimit;
      }

      if (!(leftEdge > -1.7976931348623157E+308)) {
        leftEdge = -1.7976931348623157E+308;
      }

      if (nbinsActual > 1) {
        epsxScale = xHighLimit - leftEdge;
        xScale = epsxScale / (double)nbinsActual;
        epsxScale = rt_powd_snf(10.0, floor(log10(epsxScale / ((double)
          nbinsActual - 1.0) - xScale)));
        binWidth = epsxScale * ceil(xScale / epsxScale);
      }

      u0 = leftEdge + (double)nbinsActual * binWidth;
      if ((u0 > xHighLimit) || rtIsNaN(xHighLimit)) {
        xScale = u0;
      } else {
        xScale = xHighLimit;
      }

      if (!(xScale < 1.7976931348623157E+308)) {
        xScale = 1.7976931348623157E+308;
      }
    } else {
      epsxScale = (double)nbinsActual * eps(xScale);
      epsxScale = ceil(epsxScale);
      if ((1.0 > epsxScale) || rtIsNaN(epsxScale)) {
        epsxScale = 1.0;
      }

      leftEdge = floor(2.0 * (xLowLimit - epsxScale / 4.0)) / 2.0;
      xScale = ceil(2.0 * (xHighLimit + epsxScale / 4.0)) / 2.0;
      binWidth = (xScale - leftEdge) / (double)nbinsActual;
    }

    if ((!rtIsInf(binWidth)) && (!rtIsNaN(binWidth))) {
      low_ip1 = edges->size[0] * edges->size[1];
      edges->size[0] = 1;
      edges->size[1] = nbinsActual + 1;
      emxEnsureCapacity_real_T(edges, low_ip1);
      high_i = nbinsActual + 1;
      for (low_ip1 = 0; low_ip1 < high_i; low_ip1++) {
        edges->data[low_ip1] = 0.0;
      }

      edges->data[0] = leftEdge;
      for (high_i = 0; high_i <= nbinsActual - 2; high_i++) {
        edges->data[1 + high_i] = leftEdge + (1.0 + (double)high_i) * binWidth;
      }

      edges->data[edges->size[1] - 1] = xScale;
    } else {
      low_ip1 = edges->size[0] * edges->size[1];
      edges->size[0] = 1;
      edges->size[1] = nbinsActual + 1;
      emxEnsureCapacity_real_T(edges, low_ip1);
      edges->data[edges->size[1] - 1] = xScale;
      if (edges->size[1] >= 2) {
        edges->data[0] = leftEdge;
        if (edges->size[1] >= 3) {
          if (((leftEdge < 0.0) != (xScale < 0.0)) && ((fabs(leftEdge) >
                8.9884656743115785E+307) || (fabs(xScale) >
                8.9884656743115785E+307))) {
            binWidth = leftEdge / ((double)edges->size[1] - 1.0);
            epsxScale = xScale / ((double)edges->size[1] - 1.0);
            low_ip1 = edges->size[1];
            for (k = 0; k <= low_ip1 - 3; k++) {
              edges->data[1 + k] = (leftEdge + epsxScale * (1.0 + (double)k)) -
                binWidth * (1.0 + (double)k);
            }
          } else {
            binWidth = (xScale - leftEdge) / ((double)edges->size[1] - 1.0);
            low_ip1 = edges->size[1];
            for (k = 0; k <= low_ip1 - 3; k++) {
              edges->data[1 + k] = leftEdge + (1.0 + (double)k) * binWidth;
            }
          }
        }
      }
    }
  } else if (nbinsActual >= 2) {
    low_ip1 = edges->size[0] * edges->size[1];
    edges->size[0] = 1;
    edges->size[1] = nbinsActual + 1;
    emxEnsureCapacity_real_T(edges, low_ip1);
    high_i = nbinsActual + 1;
    for (low_ip1 = 0; low_ip1 < high_i; low_ip1++) {
      edges->data[low_ip1] = 0.0;
    }

    for (k = 0; k <= nbinsActual; k++) {
      edges->data[k] = k;
    }
  } else {
    low_ip1 = edges->size[0] * edges->size[1];
    edges->size[0] = 1;
    edges->size[1] = 2;
    emxEnsureCapacity_real_T(edges, low_ip1);
    edges->data[0] = 0.0;
    edges->data[1] = 1.0;
  }

  emxInit_int32_T(&ni, 2);
  low_ip1 = ni->size[0] * ni->size[1];
  ni->size[0] = 1;
  ni->size[1] = edges->size[1] - 1;
  emxEnsureCapacity_int32_T(ni, low_ip1);
  high_i = edges->size[1] - 1;
  for (low_ip1 = 0; low_ip1 < high_i; low_ip1++) {
    ni->data[low_ip1] = 0;
  }

  nx = x->size[0];
  leftEdge = edges->data[0];
  epsxScale = edges->data[1] - edges->data[0];
  for (k = 0; k < nx; k++) {
    if ((x->data[k] >= leftEdge) && (x->data[k] <= edges->data[edges->size[1] -
         1])) {
      xScale = ceil((x->data[k] - leftEdge) / epsxScale);
      guard1 = false;
      if ((xScale >= 1.0) && (xScale < edges->size[1])) {
        low_ip1 = (int)xScale - 1;
        if ((x->data[k] >= edges->data[low_ip1]) && (x->data[k] < edges->data
             [(int)xScale])) {
          ni->data[low_ip1]++;
        } else {
          guard1 = true;
        }
      } else {
        guard1 = true;
      }

      if (guard1) {
        high_i = edges->size[1];
        low_i = 1;
        low_ip1 = 2;
        while (high_i > low_ip1) {
          nbinsActual = (low_i >> 1) + (high_i >> 1);
          if (((low_i & 1) == 1) && ((high_i & 1) == 1)) {
            nbinsActual++;
          }

          if (x->data[k] >= edges->data[nbinsActual - 1]) {
            low_i = nbinsActual;
            low_ip1 = nbinsActual + 1;
          } else {
            high_i = nbinsActual;
          }
        }

        ni->data[low_i - 1]++;
      }
    }
  }

  low_ip1 = n->size[0] * n->size[1];
  n->size[0] = 1;
  n->size[1] = ni->size[1];
  emxEnsureCapacity_real_T(n, low_ip1);
  high_i = ni->size[0] * ni->size[1];
  for (low_ip1 = 0; low_ip1 < high_i; low_ip1++) {
    n->data[low_ip1] = ni->data[low_ip1];
  }

  emxFree_int32_T(&ni);
}

/*
 * File trailer for histcounts.c
 *
 * [EOF]
 */
