/*
 *
 * gmm_uborder_fun.c
 *
 * Code generation for function 'gmm_uborder_fun'
 *
 */

/* Include files */
#include "gmm_uborder_fun.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
double __anon_fcn(const double mu[2], const double sig[2], const double amp[2],
                  double x)
{
  double varargout_1;
  double varargin_1[2];
  normpdfs(x, mu, sig, amp, varargin_1);
  if ((varargin_1[0] < varargin_1[1]) || (rtIsNaN(varargin_1[0]) && (!rtIsNaN
        (varargin_1[1])))) {
    varargout_1 = varargin_1[1];
  } else {
    varargout_1 = varargin_1[0];
  }

  return varargout_1;
}

/*
 * function [ vals ] = normpdfs(x, mu, sig, amp)
 */
void b_normpdfs(double x, const emxArray_real_T *mu, const emxArray_real_T *sig,
                const emxArray_real_T *amp, emxArray_real_T *vals)
{
  int ub_loop;
  int loop_ub;
  int i;
  double t;

  /* 'gmm_uborder_fun:14' n = length(mu); */
  /* 'gmm_uborder_fun:15' vals = NaN(n, length(x)); */
  ub_loop = vals->size[0];
  vals->size[0] = mu->size[0];
  emxEnsureCapacity_real_T(vals, ub_loop);
  loop_ub = mu->size[0];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    vals->data[ub_loop] = rtNaN;
  }

  ub_loop = mu->size[0] - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(t)

  for (i = 0; i <= ub_loop; i++) {
    /* 'gmm_uborder_fun:17' vals(i, :) = amp(i) * normpdf(x, mu(i), sig(i)); */
    if (sig->data[i] > 0.0) {
      t = (x - mu->data[i]) / sig->data[i];
      t = exp(-0.5 * t * t) / (2.5066282746310002 * sig->data[i]);
    } else {
      t = rtNaN;
    }

    vals->data[i] = amp->data[i] * t;
  }
}

/*
 * function [ vals ] = normpdfs(x, mu, sig, amp)
 */
void normpdfs(double x, const double mu[2], const double sig[2], const double
              amp[2], double vals[2])
{
  int i;
  double t;

  /* 'gmm_uborder_fun:14' n = length(mu); */
  /* 'gmm_uborder_fun:15' vals = NaN(n, length(x)); */
#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(t)

  for (i = 0; i < 2; i++) {
    /* 'gmm_uborder_fun:17' vals(i, :) = amp(i) * normpdf(x, mu(i), sig(i)); */
    if (sig[i] > 0.0) {
      t = (x - mu[i]) / sig[i];
      t = exp(-0.5 * t * t) / (2.5066282746310002 * sig[i]);
    } else {
      t = rtNaN;
    }

    vals[i] = amp[i] * t;
  }
}

/* End of code generation (gmm_uborder_fun.c) */
