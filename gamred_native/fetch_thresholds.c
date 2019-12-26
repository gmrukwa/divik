/*
 * File: fetch_thresholds.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "sort1.h"
#include "unaryMinOrMax.h"
#include "gmm_uborder_fun.h"
#include "fminbnd.h"
#include "gaussian_mixture_simple.h"

/* Function Definitions */

/*
 * function [ thresholds ] = fetch_thresholds( vals, max_components_number )
 * FETCH_THRESHOLDS Decomposes parameters into Gaussian peaks and finds
 * crossings
 *    thresholds = FETCH_THRESHOLDS(vals, max_components_number) - returns
 *    cell containing all thresholds. If no crossings are present,
 *    no threshold will be returned.
 * Arguments    : const emxArray_real_T *vals
 *                unsigned long max_components_number
 *                emxArray_real_T *thresholds
 * Return Type  : void
 */
void fetch_thresholds(const emxArray_real_T *vals, unsigned long
                      max_components_number, emxArray_real_T *thresholds)
{
  coder_internal_ref_1 lobj_2;
  coder_internal_ref_1 lobj_1;
  coder_internal_ref_1 lobj_0;
  emxArray_real_T *BIC;
  int n;
  int i0;
  int idx;
  emxArray_cell_wrap_0 *a;
  emxArray_cell_wrap_0 *mu;
  emxArray_cell_wrap_0 *sig;
  emxArray_real_T *TIC;
  emxArray_real_T *best_mu;
  int compnum;
  double t;
  int k;
  boolean_T exitg1;
  emxArray_boolean_T *x;
  int ii_data[1];
  unsigned int best_compnum__data[1];
  int best_compnum_tmp;
  coder_internal_ref_1 *iobj_0;
  coder_internal_ref_1 *iobj_1;
  coder_internal_ref_1 *iobj_2;
  emxArray_real_T *out_;
  int j;
  int i1;
  unsigned int b_k;
  double mu_idx_0;
  double crossing;
  double sig_idx_1;
  double amp_idx_0;
  double amp_idx_1;
  coder_internal_ref *b_iobj_0;
  coder_internal_ref lobj_3;
  coder_internal_ref *b_iobj_1;
  coder_internal_ref lobj_4;
  coder_internal_ref *b_iobj_2;
  coder_internal_ref lobj_5;
  coder_internal_ref *g_tunableEnvironment[3];
  c_emxInitStruct_coder_internal_(&lobj_2);
  c_emxInitStruct_coder_internal_(&lobj_1);
  c_emxInitStruct_coder_internal_(&lobj_0);
  emxInit_real_T(&BIC, 2);

  /* 'fetch_thresholds:8' n = size(vals(:), 1); */
  n = vals->size[0];

  /* 'fetch_thresholds:10' thresholds = []; */
  thresholds->size[0] = 0;

  /* 'fetch_thresholds:12' max_components_number = double(max_components_number); */
  /* 'fetch_thresholds:14' a = cell(1,max_components_number); */
  /* 'fetch_thresholds:15' mu = cell(1,max_components_number); */
  /* 'fetch_thresholds:16' sig = cell(1,max_components_number); */
  /* 'fetch_thresholds:17' TIC = cell(1,max_components_number); */
  /* 'fetch_thresholds:18' BIC = Inf*ones(1,max_components_number); */
  i0 = BIC->size[0] * BIC->size[1];
  BIC->size[0] = 1;
  idx = (int)(double)max_components_number;
  BIC->size[1] = idx;
  emxEnsureCapacity_real_T(BIC, i0);
  for (i0 = 0; i0 < idx; i0++) {
    BIC->data[i0] = rtInf;
  }

  emxInit_cell_wrap_0(&a, 2);
  emxInit_cell_wrap_0(&mu, 2);
  emxInit_cell_wrap_0(&sig, 2);
  emxInit_real_T(&TIC, 2);

  /* 'fetch_thresholds:19' param = vals(:); */
  i0 = sig->size[0] * sig->size[1];
  sig->size[0] = 1;
  sig->size[1] = idx;
  emxEnsureCapacity_cell_wrap_0(sig, i0);
  i0 = mu->size[0] * mu->size[1];
  mu->size[0] = 1;
  mu->size[1] = idx;
  emxEnsureCapacity_cell_wrap_0(mu, i0);
  i0 = a->size[0] * a->size[1];
  a->size[0] = 1;
  a->size[1] = idx;
  emxEnsureCapacity_cell_wrap_0(a, i0);
  i0 = TIC->size[0] * TIC->size[1];
  TIC->size[0] = 1;
  TIC->size[1] = idx;
  emxEnsureCapacity_real_T(TIC, i0);
  emxInit_real_T(&best_mu, 1);
  for (compnum = 0; compnum < idx; compnum++) {
    /* 'fetch_thresholds:21' [a{compnum},mu{compnum},sig{compnum},TIC{compnum},l_lik]=gaussian_mixture_simple(param,ones(n,1),compnum); */
    i0 = best_mu->size[0];
    best_mu->size[0] = n;
    emxEnsureCapacity_real_T(best_mu, i0);
    for (i0 = 0; i0 < n; i0++) {
      best_mu->data[i0] = 1.0;
    }

    gaussian_mixture_simple(vals, best_mu, 1.0 + (double)compnum, a->
      data[compnum].f1, mu->data[compnum].f1, sig->data[compnum].f1, &TIC->
      data[compnum], &t);

    /* 'fetch_thresholds:22' BIC(compnum) = -2*l_lik+(3*compnum-1)*log(n); */
    BIC->data[compnum] = -2.0 * t + (3.0 * (1.0 + (double)compnum) - 1.0) * log
      (n);
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
      i0 = idx + 1;
      for (k = i0; k <= n; k++) {
        if (t > BIC->data[k - 1]) {
          t = BIC->data[k - 1];
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
        i0 = idx + 1;
        for (k = i0; k <= n; k++) {
          if (t > BIC->data[k - 1]) {
            t = BIC->data[k - 1];
          }
        }
      }
    }

    emxInit_boolean_T(&x, 2);
    i0 = x->size[0] * x->size[1];
    x->size[0] = 1;
    x->size[1] = BIC->size[1];
    emxEnsureCapacity_boolean_T(x, i0);
    idx = BIC->size[0] * BIC->size[1];
    for (i0 = 0; i0 < idx; i0++) {
      x->data[i0] = (BIC->data[i0] == t);
    }

    k = (1 <= x->size[1]);
    idx = 0;
    compnum = 0;
    exitg1 = false;
    while ((!exitg1) && (compnum <= x->size[1] - 1)) {
      if (x->data[compnum]) {
        idx = 1;
        ii_data[0] = compnum + 1;
        exitg1 = true;
      } else {
        compnum++;
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

    for (i0 = 0; i0 < k; i0++) {
      best_compnum__data[0] = (unsigned int)ii_data[0];
    }

    /* 'fetch_thresholds:30' best_compnum = best_compnum_(1); */
    best_compnum_tmp = (int)best_compnum__data[0] - 1;

    /* 'fetch_thresholds:31' best_mu = mu{best_compnum}; */
    i0 = best_mu->size[0];
    best_mu->size[0] = mu->data[best_compnum_tmp].f1->size[0];
    emxEnsureCapacity_real_T(best_mu, i0);
    idx = mu->data[best_compnum_tmp].f1->size[0];
    for (i0 = 0; i0 < idx; i0++) {
      best_mu->data[i0] = mu->data[best_compnum_tmp].f1->data[i0];
    }

    /* 'fetch_thresholds:32' best_sig = sig{best_compnum}; */
    /* 'fetch_thresholds:33' best_a = a{best_compnum}; */
    /* FIX */
    /* 'fetch_thresholds:35' TIC = TIC{best_compnum}; */
    /* 'fetch_thresholds:37' f = gmm_uborder_fun(best_mu, best_sig, best_a); */
    iobj_0 = &lobj_0;
    iobj_1 = &lobj_1;
    iobj_2 = &lobj_2;

    /* GMM_UBORDER_FUN Finds the function which represents GMM. */
    /*    f = GMM_UBORDER_FUN(mu,sig,amp) - returns a handle to function (f) */
    /*    which calculates value of the upper border of peaks specified by its */
    /*    mean (mu), standard deviation (sig) and amplitude (amp). Parameters: */
    /* 	mu, sig, amp can be at max two dimensional. */
    /* 'gmm_uborder_fun:8' if sum(size(mu)==size(sig))<length(size(mu)) || sum(size(mu)==size(amp))<length(size(mu)) */
    /* 'gmm_uborder_fun:12' mu_ = mu(:); */
    i0 = lobj_0.contents->size[0];
    lobj_0.contents->size[0] = mu->data[best_compnum_tmp].f1->size[0];
    emxEnsureCapacity_real_T(lobj_0.contents, i0);
    idx = mu->data[best_compnum_tmp].f1->size[0];
    for (i0 = 0; i0 < idx; i0++) {
      lobj_0.contents->data[i0] = mu->data[best_compnum_tmp].f1->data[i0];
    }

    /* 'gmm_uborder_fun:13' sig_ = sig(:); */
    i0 = lobj_1.contents->size[0];
    lobj_1.contents->size[0] = sig->data[best_compnum_tmp].f1->size[0];
    emxEnsureCapacity_real_T(lobj_1.contents, i0);
    idx = sig->data[best_compnum_tmp].f1->size[0];
    for (i0 = 0; i0 < idx; i0++) {
      lobj_1.contents->data[i0] = sig->data[best_compnum_tmp].f1->data[i0];
    }

    /* 'gmm_uborder_fun:14' amp_ = amp(:); */
    i0 = lobj_2.contents->size[0];
    lobj_2.contents->size[0] = a->data[best_compnum_tmp].f1->size[0];
    emxEnsureCapacity_real_T(lobj_2.contents, i0);
    idx = a->data[best_compnum_tmp].f1->size[0];
    for (i0 = 0; i0 < idx; i0++) {
      lobj_2.contents->data[i0] = a->data[best_compnum_tmp].f1->data[i0];
    }

    /* 'gmm_uborder_fun:28' f = @fun; */
    /*  	[n,m] = size(mu); */
    /*  	 */
    /*  	f = @(x) -Inf; */
    /*  	for i=1:n */
    /*  		for j=1:m */
    /*  			f = @(x) max(cat(3,amp(i,j)*normpdf(x,mu(i,j),sig(i,j)),f(x)),[],3); */
    /*  		end */
    /*  	end */
    /* if there are few components */
    /* 'fetch_thresholds:40' if best_compnum>1 */
    if ((int)best_compnum__data[0] > 1) {
      /* they are picked pairwise */
      /* 'fetch_thresholds:42' for j=1:(best_compnum-1) */
      i0 = (int)best_compnum__data[0];
      emxInit_real_T(&out_, 1);
      for (j = 0; j <= i0 - 2; j++) {
        /* 'fetch_thresholds:43' for k=(j+1):best_compnum */
        i1 = best_compnum_tmp - j;
        for (k = 0; k < i1; k++) {
          b_k = ((unsigned int)j + k) + 2U;

          /* their upper bound */
          /* 'fetch_thresholds:45' g = gmm_uborder_fun(best_mu([j,k]), best_sig([j,k]), best_a([j,k])); */
          mu_idx_0 = mu->data[best_compnum_tmp].f1->data[j];
          t = mu->data[best_compnum_tmp].f1->data[(int)b_k - 1];
          crossing = sig->data[best_compnum_tmp].f1->data[j];
          sig_idx_1 = sig->data[best_compnum_tmp].f1->data[(int)b_k - 1];
          amp_idx_0 = a->data[best_compnum_tmp].f1->data[j];
          amp_idx_1 = a->data[best_compnum_tmp].f1->data[(int)b_k - 1];
          b_iobj_0 = &lobj_3;
          b_iobj_1 = &lobj_4;
          b_iobj_2 = &lobj_5;

          /* GMM_UBORDER_FUN Finds the function which represents GMM. */
          /*    f = GMM_UBORDER_FUN(mu,sig,amp) - returns a handle to function (f) */
          /*    which calculates value of the upper border of peaks specified by its */
          /*    mean (mu), standard deviation (sig) and amplitude (amp). Parameters: */
          /* 	mu, sig, amp can be at max two dimensional. */
          /* 'gmm_uborder_fun:8' if sum(size(mu)==size(sig))<length(size(mu)) || sum(size(mu)==size(amp))<length(size(mu)) */
          /* 'gmm_uborder_fun:12' mu_ = mu(:); */
          lobj_3.contents[0] = mu_idx_0;
          lobj_3.contents[1] = t;

          /* 'gmm_uborder_fun:13' sig_ = sig(:); */
          lobj_4.contents[0] = crossing;
          lobj_4.contents[1] = sig_idx_1;

          /* 'gmm_uborder_fun:14' amp_ = amp(:); */
          lobj_5.contents[0] = amp_idx_0;
          lobj_5.contents[1] = amp_idx_1;

          /* 'gmm_uborder_fun:28' f = @fun; */
          g_tunableEnvironment[0] = &lobj_3;
          g_tunableEnvironment[1] = &lobj_5;
          g_tunableEnvironment[2] = &lobj_4;

          /*  	[n,m] = size(mu); */
          /*  	 */
          /*  	f = @(x) -Inf; */
          /*  	for i=1:n */
          /*  		for j=1:m */
          /*  			f = @(x) max(cat(3,amp(i,j)*normpdf(x,mu(i,j),sig(i,j)),f(x)),[],3); */
          /*  		end */
          /*  	end */
          /* the disjunction condition & weight condition is checked */
          /* 'fetch_thresholds:47' if g(best_mu(j))==best_a(j)*normpdf(best_mu(j),best_mu(j),best_sig(j)) && ... */
          /* 'fetch_thresholds:48'                         g(best_mu(k))==best_a(k)*normpdf(best_mu(k),best_mu(k),best_sig(k)) */
          mu_idx_0 = fun(b_iobj_0, b_iobj_2, b_iobj_1, mu->data[(int)
                         best_compnum__data[0] - 1].f1->data[j]);
          if (sig->data[best_compnum_tmp].f1->data[j] > 0.0) {
            t = (mu->data[best_compnum_tmp].f1->data[j] - mu->
                 data[best_compnum_tmp].f1->data[j]) / sig->
              data[best_compnum_tmp].f1->data[j];
            t = exp(-0.5 * t * t) / (2.5066282746310002 * sig->
              data[best_compnum_tmp].f1->data[j]);
          } else {
            t = rtNaN;
          }

          if (mu_idx_0 == a->data[best_compnum_tmp].f1->data[j] * t) {
            mu_idx_0 = fun(b_iobj_0, b_iobj_2, b_iobj_1, mu->data[(int)
                           best_compnum__data[0] - 1].f1->data[(int)b_k - 1]);
            if (sig->data[best_compnum_tmp].f1->data[(int)b_k - 1] > 0.0) {
              t = (mu->data[best_compnum_tmp].f1->data[(int)b_k - 1] - mu->
                   data[best_compnum_tmp].f1->data[(int)b_k - 1]) / sig->
                data[best_compnum_tmp].f1->data[(int)b_k - 1];
              t = exp(-0.5 * t * t) / (2.5066282746310002 * sig->
                data[best_compnum_tmp].f1->data[(int)b_k - 1]);
            } else {
              t = rtNaN;
            }

            if (mu_idx_0 == a->data[best_compnum_tmp].f1->data[(int)b_k - 1] * t)
            {
              /* their crossing is found */
              /* 'fetch_thresholds:50' crossing = fminbnd(g,min(best_mu([j,k])),max((best_mu([j,k])))); */
              if ((best_mu->data[j] > best_mu->data[(int)b_k - 1]) || (rtIsNaN
                   (best_mu->data[j]) && (!rtIsNaN(best_mu->data[(int)b_k - 1]))))
              {
                t = best_mu->data[(int)b_k - 1];
              } else {
                t = best_mu->data[j];
              }

              if ((best_mu->data[j] < best_mu->data[(int)b_k - 1]) || (rtIsNaN
                   (best_mu->data[j]) && (!rtIsNaN(best_mu->data[(int)b_k - 1]))))
              {
                mu_idx_0 = best_mu->data[(int)b_k - 1];
              } else {
                mu_idx_0 = best_mu->data[j];
              }

              crossing = fminbnd(g_tunableEnvironment, t, mu_idx_0);

              /* and it is checked to be on the upper border */
              /* 'fetch_thresholds:52' if f(crossing)==g(crossing) */
              /* 'gmm_uborder_fun:18' in_ = in(:); */
              /* 'gmm_uborder_fun:19' n_comp = size(mu_, 1); */
              idx = lobj_0.contents->size[0] - 1;

              /* 'gmm_uborder_fun:20' out_ = NaN(n_comp, size(in_, 1)); */
              n = out_->size[0];
              out_->size[0] = idx + 1;
              emxEnsureCapacity_real_T(out_, n);
              for (n = 0; n <= idx; n++) {
                out_->data[n] = rtNaN;
              }

              /* 'gmm_uborder_fun:21' for i=1:n_comp */
              for (compnum = 0; compnum <= idx; compnum++) {
                /* 'gmm_uborder_fun:22' out_(i, :) = amp_(i) * normpdf(in_, mu_(i), sig_(i)); */
                t = iobj_0->contents->data[compnum];
                mu_idx_0 = iobj_1->contents->data[compnum];
                if (mu_idx_0 > 0.0) {
                  t = (crossing - t) / mu_idx_0;
                  t = exp(-0.5 * t * t) / (2.5066282746310002 * mu_idx_0);
                } else {
                  t = rtNaN;
                }

                out_->data[compnum] = iobj_2->contents->data[compnum] * t;
              }

              /* 'gmm_uborder_fun:24' out = max(out_, [], 1); */
              mu_idx_0 = fun(b_iobj_0, b_iobj_2, b_iobj_1, crossing);
              if (minOrMaxRealFloatVector(out_) == mu_idx_0) {
                /* if it truly is, then it is saved */
                /* 'fetch_thresholds:54' thresholds = [thresholds; crossing]; */
                n = thresholds->size[0];
                compnum = thresholds->size[0];
                thresholds->size[0] = n + 1;
                emxEnsureCapacity_real_T(thresholds, compnum);
                thresholds->data[n] = crossing;
              }
            }
          }
        }
      }

      emxFree_real_T(&out_);

      /* 'fetch_thresholds:59' thresholds = sort(thresholds,'ascend'); */
      b_sort(thresholds);
    }
  }

  emxFree_real_T(&best_mu);
  emxFree_real_T(&BIC);
  emxFree_cell_wrap_0(&sig);
  emxFree_cell_wrap_0(&mu);
  emxFree_cell_wrap_0(&a);
  c_emxFreeStruct_coder_internal_(&lobj_0);
  c_emxFreeStruct_coder_internal_(&lobj_1);
  c_emxFreeStruct_coder_internal_(&lobj_2);
}

/*
 * File trailer for fetch_thresholds.c
 *
 * [EOF]
 */
