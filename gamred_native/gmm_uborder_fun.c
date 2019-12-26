/*
 * File: gmm_uborder_fun.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "gmm_uborder_fun.h"

/* Function Definitions */

/*
 * function [ out ] = fun( in )
 * Arguments    : const coder_internal_ref *mu_
 *                const coder_internal_ref *amp_
 *                const coder_internal_ref *sig_
 *                double in
 * Return Type  : double
 */
double fun(const coder_internal_ref *mu_, const coder_internal_ref *amp_, const
           coder_internal_ref *sig_, double in)
{
  double out;
  double t;
  double sigma;

  /* 'gmm_uborder_fun:18' in_ = in(:); */
  /* 'gmm_uborder_fun:19' n_comp = size(mu_, 1); */
  /* 'gmm_uborder_fun:20' out_ = NaN(n_comp, size(in_, 1)); */
  /* 'gmm_uborder_fun:21' for i=1:n_comp */
  /* 'gmm_uborder_fun:22' out_(i, :) = amp_(i) * normpdf(in_, mu_(i), sig_(i)); */
  t = mu_->contents[0];
  sigma = sig_->contents[0];
  if (sigma > 0.0) {
    t = (in - t) / sigma;
    t = exp(-0.5 * t * t) / (2.5066282746310002 * sigma);
  } else {
    t = rtNaN;
  }

  out = amp_->contents[0] * t;

  /* 'gmm_uborder_fun:22' out_(i, :) = amp_(i) * normpdf(in_, mu_(i), sig_(i)); */
  t = mu_->contents[1];
  sigma = sig_->contents[1];
  if (sigma > 0.0) {
    t = (in - t) / sigma;
    t = exp(-0.5 * t * t) / (2.5066282746310002 * sigma);
  } else {
    t = rtNaN;
  }

  t *= amp_->contents[1];

  /* 'gmm_uborder_fun:24' out = max(out_, [], 1); */
  if ((out < t) || (rtIsNaN(out) && (!rtIsNaN(t)))) {
    out = t;
  }

  return out;
}

/*
 * File trailer for gmm_uborder_fun.c
 *
 * [EOF]
 */
