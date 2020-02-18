/*
 *
 * fetch_thresholds.c
 *
 * Code generation for function 'fetch_thresholds'
 *
 */

/* Include files */
#include "fetch_thresholds.h"
#include "fetch_thresholds_data.h"
#include "fetch_thresholds_emxutil.h"
#include "fetch_thresholds_initialize.h"
#include "fminbnd.h"
#include "gaussian_mixture_simple.h"
#include "gmm_uborder_fun.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include <math.h>

/* Function Definitions */

/*
 * function [ thresholds ] = fetch_thresholds( vals, max_components_number )
 */
void fetch_thresholds(const emxArray_real_T *vals, unsigned long
                      max_components_number, emxArray_real_T *thresholds)
{
  emxArray_cell_wrap_0 *a;
  emxArray_real_T *BIC;
  int n;
  int i;
  int loop_ub;
  emxArray_real_T *param;
  int idx;
  emxArray_cell_wrap_0 *mu;
  emxArray_cell_wrap_0 *sig;
  emxArray_real_T *TIC;
  int compnum;
  emxArray_real_T *r;
  double l_lik;
  double t;
  int k;
  bool exitg1;
  double d;
  int i1;
  emxArray_boolean_T *x;
  int i_data[1];
  unsigned int best_compnum__data[1];
  int best_compnum_tmp;
  int j;
  int i2;
  double y;
  double b_mu[2];
  double b_sig[2];
  double amp[2];
  cell_wrap_2 g_tunableEnvironment[3];
  double crossing;
  if (isInitialized_fetch_thresholds == false) {
    fetch_thresholds_initialize();
  }

  emxInit_cell_wrap_0(&a, 2);
  emxInit_real_T(&BIC, 2);

  /* FETCH_THRESHOLDS Decomposes parameters into Gaussian peaks and finds */
  /* crossings */
  /*    thresholds = FETCH_THRESHOLDS(vals, max_components_number) - returns */
  /*    cell containing all thresholds. If no crossings are present, */
  /*    no threshold will be returned. */
  /* 'fetch_thresholds:8' n = size(vals(:), 1); */
  n = vals->size[0];

  /* 'fetch_thresholds:10' thresholds = []; */
  thresholds->size[0] = 0;

  /* 'fetch_thresholds:12' max_components_number = double(max_components_number); */
  /* 'fetch_thresholds:14' a = cell(1,max_components_number); */
  i = a->size[0] * a->size[1];
  a->size[0] = 1;
  loop_ub = (int)(double)max_components_number;
  a->size[1] = loop_ub;
  emxEnsureCapacity_cell_wrap_0(a, i);

  /* 'fetch_thresholds:15' mu = cell(1,max_components_number); */
  /* 'fetch_thresholds:16' sig = cell(1,max_components_number); */
  /* 'fetch_thresholds:17' TIC = cell(1,max_components_number); */
  /* 'fetch_thresholds:18' BIC = Inf*ones(1,max_components_number); */
  i = BIC->size[0] * BIC->size[1];
  BIC->size[0] = 1;
  BIC->size[1] = loop_ub;
  emxEnsureCapacity_real_T(BIC, i);
  for (i = 0; i < loop_ub; i++) {
    BIC->data[i] = rtInf;
  }

  emxInit_real_T(&param, 1);

  /* 'fetch_thresholds:19' param = vals(:); */
  i = param->size[0];
  param->size[0] = vals->size[0];
  emxEnsureCapacity_real_T(param, i);
  idx = vals->size[0];
  for (i = 0; i < idx; i++) {
    param->data[i] = vals->data[i];
  }

  emxInit_cell_wrap_0(&mu, 2);
  emxInit_cell_wrap_0(&sig, 2);
  emxInit_real_T(&TIC, 2);
  i = sig->size[0] * sig->size[1];
  sig->size[0] = 1;
  sig->size[1] = loop_ub;
  emxEnsureCapacity_cell_wrap_0(sig, i);
  i = mu->size[0] * mu->size[1];
  mu->size[0] = 1;
  mu->size[1] = loop_ub;
  emxEnsureCapacity_cell_wrap_0(mu, i);
  i = TIC->size[0] * TIC->size[1];
  TIC->size[0] = 1;
  TIC->size[1] = loop_ub;
  emxEnsureCapacity_real_T(TIC, i);
  loop_ub--;

#pragma omp parallel \
 num_threads(omp_get_max_threads()) \
 private(r,l_lik,i1)

  {
    emxInit_real_T(&r, 1);

#pragma omp for nowait

    for (compnum = 0; compnum <= loop_ub; compnum++) {
      /* 'fetch_thresholds:21' [a{compnum},mu{compnum},sig{compnum},TIC{compnum},l_lik]=gaussian_mixture_simple(param,ones(n,1),compnum); */
      i1 = r->size[0];
      r->size[0] = n;
      emxEnsureCapacity_real_T(r, i1);
      for (i1 = 0; i1 < n; i1++) {
        r->data[i1] = 1.0;
      }

      gaussian_mixture_simple(param, r, (double)compnum + 1.0, a->data[compnum].
        f1, mu->data[compnum].f1, sig->data[compnum].f1, &TIC->data[compnum],
        &l_lik);

      /* 'fetch_thresholds:22' BIC(compnum) = -2*l_lik+(3*compnum-1)*log(n); */
      BIC->data[compnum] = -2.0 * l_lik + (3.0 * ((double)compnum + 1.0) - 1.0) *
        log(n);
    }

    emxFree_real_T(&r);
  }

  emxFree_real_T(&TIC);

  /* 'fetch_thresholds:25' if isinf(min(BIC)) */
  n = BIC->size[1];
  if (BIC->size[1] <= 2) {
    if (BIC->size[1] == 1) {
      t = BIC->data[0];
    } else if ((BIC->data[0] > BIC->data[1]) || (rtIsNaN(BIC->data[0]) &&
                (!rtIsNaN(BIC->data[1])))) {
      t = BIC->data[1];
    } else {
      t = BIC->data[0];
    }
  } else {
    if (!rtIsNaN(BIC->data[0])) {
      idx = 1;
    } else {
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= BIC->size[1])) {
        if (!rtIsNaN(BIC->data[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }

    if (idx == 0) {
      t = BIC->data[0];
    } else {
      t = BIC->data[idx - 1];
      i = idx + 1;
      for (k = i; k <= n; k++) {
        d = BIC->data[k - 1];
        if (t > d) {
          t = d;
        }
      }
    }
  }

  if (!rtIsInf(t)) {
    /* 'fetch_thresholds:29' best_compnum_ = find(BIC == min(BIC),1); */
    n = BIC->size[1];
    if (BIC->size[1] <= 2) {
      if (BIC->size[1] == 1) {
        t = BIC->data[0];
      } else if ((BIC->data[0] > BIC->data[1]) || (rtIsNaN(BIC->data[0]) &&
                  (!rtIsNaN(BIC->data[1])))) {
        t = BIC->data[1];
      } else {
        t = BIC->data[0];
      }
    } else {
      if (!rtIsNaN(BIC->data[0])) {
        idx = 1;
      } else {
        idx = 0;
        k = 2;
        exitg1 = false;
        while ((!exitg1) && (k <= BIC->size[1])) {
          if (!rtIsNaN(BIC->data[k - 1])) {
            idx = k;
            exitg1 = true;
          } else {
            k++;
          }
        }
      }

      if (idx == 0) {
        t = BIC->data[0];
      } else {
        t = BIC->data[idx - 1];
        i = idx + 1;
        for (k = i; k <= n; k++) {
          d = BIC->data[k - 1];
          if (t > d) {
            t = d;
          }
        }
      }
    }

    emxInit_boolean_T(&x, 2);
    i = x->size[0] * x->size[1];
    x->size[0] = 1;
    x->size[1] = BIC->size[1];
    emxEnsureCapacity_boolean_T(x, i);
    loop_ub = BIC->size[0] * BIC->size[1];
    for (i = 0; i < loop_ub; i++) {
      x->data[i] = (BIC->data[i] == t);
    }

    k = (1 <= x->size[1]);
    idx = 0;
    loop_ub = 0;
    exitg1 = false;
    while ((!exitg1) && (loop_ub <= x->size[1] - 1)) {
      if (x->data[loop_ub]) {
        idx = 1;
        i_data[0] = loop_ub + 1;
        exitg1 = true;
      } else {
        loop_ub++;
      }
    }

    emxFree_boolean_T(&x);
    if (k == 1) {
      if (idx == 0) {
        k = 0;
      }
    } else {
      k = (1 <= idx);
    }

    if (0 <= k - 1) {
      best_compnum__data[0] = (unsigned int)i_data[0];
    }

    /* 'fetch_thresholds:30' best_compnum = best_compnum_(1); */
    best_compnum_tmp = (int)best_compnum__data[0] - 1;

    /* 'fetch_thresholds:31' best_mu = mu{best_compnum}; */
    /* 'fetch_thresholds:32' best_sig = sig{best_compnum}; */
    /* 'fetch_thresholds:33' best_a = a{best_compnum}; */
    /* FIX */
    /* 'fetch_thresholds:35' TIC = TIC{best_compnum}; */
    /* 'fetch_thresholds:37' f = gmm_uborder_fun(best_mu, best_sig, best_a); */
    /* GMM_UBORDER_FUN Finds the function which represents GMM. */
    /*    f = GMM_UBORDER_FUN(mu,sig,amp) - returns a handle to function (f) */
    /*    which calculates value of the upper border of peaks specified by its */
    /*    mean (mu), standard deviation (sig) and amplitude (amp). */
    /* 'gmm_uborder_fun:7' f = @(x) max(normpdfs(x, mu(:), sig(:), amp(:)), [], 1); */
    /* if there are few components */
    /* 'fetch_thresholds:40' if best_compnum>1 */
    if ((int)best_compnum__data[0] > 1) {
      /* they are picked pairwise */
      /* 'fetch_thresholds:42' for j=1:(best_compnum-1) */
      i = (int)best_compnum__data[0];
      for (j = 0; j <= i - 2; j++) {
        /* 'fetch_thresholds:43' for k=(j+1):best_compnum */
        i2 = best_compnum_tmp - j;
        if (0 <= i2 - 1) {
          if (sig->data[best_compnum_tmp].f1->data[j] > 0.0) {
            t = (mu->data[best_compnum_tmp].f1->data[j] - mu->
                 data[best_compnum_tmp].f1->data[j]) / sig->
              data[best_compnum_tmp].f1->data[j];
            y = exp(-0.5 * t * t) / (2.5066282746310002 * sig->
              data[best_compnum_tmp].f1->data[j]);
          } else {
            y = rtNaN;
          }
        }

        for (k = 0; k < i2; k++) {
          /* their upper bound */
          /* 'fetch_thresholds:45' g = gmm_uborder_fun(best_mu([j,k]), best_sig([j,k]), best_a([j,k])); */
          loop_ub = (int)((unsigned int)j + k) + 1;
          b_mu[0] = mu->data[best_compnum_tmp].f1->data[j];
          b_mu[1] = mu->data[best_compnum_tmp].f1->data[loop_ub];
          b_sig[0] = sig->data[best_compnum_tmp].f1->data[j];
          b_sig[1] = sig->data[best_compnum_tmp].f1->data[loop_ub];

          /* GMM_UBORDER_FUN Finds the function which represents GMM. */
          /*    f = GMM_UBORDER_FUN(mu,sig,amp) - returns a handle to function (f) */
          /*    which calculates value of the upper border of peaks specified by its */
          /*    mean (mu), standard deviation (sig) and amplitude (amp). */
          /* 'gmm_uborder_fun:7' f = @(x) max(normpdfs(x, mu(:), sig(:), amp(:)), [], 1); */
          amp[0] = a->data[best_compnum_tmp].f1->data[j];
          g_tunableEnvironment[0].f1[0] = mu->data[best_compnum_tmp].f1->data[j];
          g_tunableEnvironment[1].f1[0] = sig->data[best_compnum_tmp].f1->data[j];
          g_tunableEnvironment[2].f1[0] = a->data[best_compnum_tmp].f1->data[j];
          amp[1] = a->data[best_compnum_tmp].f1->data[loop_ub];
          g_tunableEnvironment[0].f1[1] = mu->data[best_compnum_tmp].f1->
            data[loop_ub];
          g_tunableEnvironment[1].f1[1] = sig->data[best_compnum_tmp].f1->
            data[loop_ub];
          g_tunableEnvironment[2].f1[1] = a->data[best_compnum_tmp].f1->
            data[loop_ub];

          /* the disjunction condition & weight condition is checked */
          /* 'fetch_thresholds:47' if g(best_mu(j))==best_a(j)*normpdf(best_mu(j),best_mu(j),best_sig(j)) && ... */
          /* 'fetch_thresholds:48'                         g(best_mu(k))==best_a(k)*normpdf(best_mu(k),best_mu(k),best_sig(k)) */
          if (__anon_fcn(b_mu, b_sig, amp, mu->data[best_compnum_tmp].f1->data[j])
              == a->data[best_compnum_tmp].f1->data[j] * y) {
            d = sig->data[(int)best_compnum__data[0] - 1].f1->data[loop_ub];
            if (d > 0.0) {
              t = (mu->data[(int)best_compnum__data[0] - 1].f1->data[loop_ub] -
                   mu->data[(int)best_compnum__data[0] - 1].f1->data[loop_ub]) /
                d;
              t = exp(-0.5 * t * t) / (2.5066282746310002 * d);
            } else {
              t = rtNaN;
            }

            if (__anon_fcn(b_mu, b_sig, amp, mu->data[best_compnum_tmp].f1->
                           data[loop_ub]) == a->data[best_compnum_tmp].f1->
                data[loop_ub] * t) {
              /* their crossing is found */
              /* 'fetch_thresholds:50' crossing = fminbnd(g,min(best_mu([j,k])),max((best_mu([j,k])))); */
              if ((mu->data[best_compnum_tmp].f1->data[j] > mu->
                   data[best_compnum_tmp].f1->data[loop_ub]) || (rtIsNaN
                   (mu->data[best_compnum_tmp].f1->data[j]) && (!rtIsNaN
                    (mu->data[best_compnum_tmp].f1->data[loop_ub])))) {
                t = mu->data[best_compnum_tmp].f1->data[loop_ub];
              } else {
                t = mu->data[best_compnum_tmp].f1->data[j];
              }

              if ((mu->data[best_compnum_tmp].f1->data[j] < mu->
                   data[best_compnum_tmp].f1->data[loop_ub]) || (rtIsNaN
                   (mu->data[best_compnum_tmp].f1->data[j]) && (!rtIsNaN
                    (mu->data[best_compnum_tmp].f1->data[loop_ub])))) {
                crossing = mu->data[best_compnum_tmp].f1->data[loop_ub];
              } else {
                crossing = mu->data[best_compnum_tmp].f1->data[j];
              }

              crossing = fminbnd(g_tunableEnvironment, t, crossing);

              /* and it is checked to be on the upper border */
              /* 'fetch_thresholds:52' if f(crossing)==g(crossing) */
              b_normpdfs(crossing, mu->data[(int)best_compnum__data[0] - 1].f1,
                         sig->data[(int)best_compnum__data[0] - 1].f1, a->data
                         [(int)best_compnum__data[0] - 1].f1, param);
              n = param->size[0];
              if (param->size[0] <= 2) {
                if (param->size[0] == 1) {
                  t = param->data[0];
                } else if ((param->data[0] < param->data[1]) || (rtIsNaN
                            (param->data[0]) && (!rtIsNaN(param->data[1])))) {
                  t = param->data[1];
                } else {
                  t = param->data[0];
                }
              } else {
                if (!rtIsNaN(param->data[0])) {
                  idx = 1;
                } else {
                  idx = 0;
                  loop_ub = 2;
                  exitg1 = false;
                  while ((!exitg1) && (loop_ub <= param->size[0])) {
                    if (!rtIsNaN(param->data[loop_ub - 1])) {
                      idx = loop_ub;
                      exitg1 = true;
                    } else {
                      loop_ub++;
                    }
                  }
                }

                if (idx == 0) {
                  t = param->data[0];
                } else {
                  t = param->data[idx - 1];
                  idx++;
                  for (loop_ub = idx; loop_ub <= n; loop_ub++) {
                    d = param->data[loop_ub - 1];
                    if (t < d) {
                      t = d;
                    }
                  }
                }
              }

              if (t == __anon_fcn(b_mu, b_sig, amp, crossing)) {
                /* if it truly is, then it is saved */
                /* 'fetch_thresholds:54' thresholds = [thresholds; crossing]; */
                idx = thresholds->size[0];
                loop_ub = thresholds->size[0];
                thresholds->size[0]++;
                emxEnsureCapacity_real_T(thresholds, loop_ub);
                thresholds->data[idx] = crossing;
              }
            }
          }
        }
      }

      /* 'fetch_thresholds:59' thresholds = sort(thresholds,'ascend'); */
      b_sort(thresholds);
    }
  }

  emxFree_real_T(&param);
  emxFree_real_T(&BIC);
  emxFree_cell_wrap_0(&sig);
  emxFree_cell_wrap_0(&mu);
  emxFree_cell_wrap_0(&a);
}

/* End of code generation (fetch_thresholds.c) */
