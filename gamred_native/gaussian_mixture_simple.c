/*
 * File: gaussian_mixture_simple.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "gaussian_mixture_simple.h"
#include "fetch_thresholds_emxutil.h"
#include "std.h"
#include "nanmean.h"
#include "g_mix_est_fast_lik.h"
#include "sum.h"
#include "my_qu_ix_w.h"
#include "dyn_pr_split_w_aux.h"
#include "mean.h"
#include "histcounts.h"
#include "sqrt.h"

/* Function Definitions */

/*
 * function [pp_est,mu_est,sig_est,TIC,l_lik]=gaussian_mixture_simple(x,counts,KS)
 * Arguments    : const emxArray_real_T *x
 *                const emxArray_real_T *counts
 *                double KS
 *                emxArray_real_T *pp_est
 *                emxArray_real_T *mu_est
 *                emxArray_real_T *sig_est
 *                double *TIC
 *                double *l_lik
 * Return Type  : void
 */
void gaussian_mixture_simple(const emxArray_real_T *x, const emxArray_real_T
  *counts, double KS, emxArray_real_T *pp_est, emxArray_real_T *mu_est,
  emxArray_real_T *sig_est, double *TIC, double *l_lik)
{
  emxArray_real_T *b_x;
  int i2;
  int loop_ub;
  double b_TIC;
  emxArray_real_T *y_out;
  emxArray_real_T *edges;
  double ex;
  double Nb;
  int i3;
  emxArray_real_T *b_edges;
  int i4;
  emxArray_real_T *mz_out;
  emxArray_real_T *aux_mx;
  emxArray_real_T *p_aux;
  int varargin_1;
  int N;
  emxArray_uint32_T *opt_pals;
  int b_loop_ub;
  emxArray_real_T *p_opt_idx;
  emxArray_real_T *b_mz_out;
  emxArray_real_T *b_y_out;
  int kk;
  emxArray_real_T *Q;
  emxArray_real_T *c_x;
  int i5;
  int kster;
  int n_tmp_tmp;
  int n_tmp;
  double b_ex;
  emxArray_real_T *part_cl;
  int k;
  boolean_T exitg1;
  emxArray_real_T *pp_ini;
  emxArray_real_T *mu_ini;
  emxArray_real_T *sig_ini;
  emxInit_real_T(&b_x, 1);

  /*  FUNCTION:  */
  /*  gaussian_mixture */
  /*  PURPOSE:  */
  /*  computes Gaussian mixture decomposition  */
  /*  the signal to be decomposed is the mass spectrum  */
  /*  INPUT: */
  /*  measurements are specified by vector x */
  /*  KS - number of Gaussian components */
  /*  METHOD:  */
  /*  EM iteriations with initial conditions for means defined by inverting the */
  /*  empirical cumulative distribution, iterations for standard deviations stabilized */
  /*  by using prior Wishart distribution */
  /*  OUTPUT: */
  /*  pp_est - vector of esimated weights */
  /*  mu_est - vector of estimated component means */
  /*  sig_est - vector of estimated standard deviations of components */
  /*  l_lik - log likelihood of the estimated decomposition */
  /*  AUTHOR: */
  /*  Andrzej Polanski */
  /*  email: andrzej.polanski@polsl.pl */
  /* 'gaussian_mixture_simple:29' TIC=sum(x.*counts); */
  i2 = b_x->size[0];
  b_x->size[0] = x->size[0];
  emxEnsureCapacity_real_T(b_x, i2);
  loop_ub = x->size[0];
  for (i2 = 0; i2 < loop_ub; i2++) {
    b_x->data[i2] = x->data[i2] * counts->data[i2];
  }

  b_TIC = sum(b_x);

  /* 'gaussian_mixture_simple:30' if KS==1 */
  emxFree_real_T(&b_x);
  if (KS == 1.0) {
    /* 'gaussian_mixture_simple:31' mu_est=nanmean(x); */
    ex = nanmean(x);
    i2 = mu_est->size[0];
    mu_est->size[0] = 1;
    emxEnsureCapacity_real_T(mu_est, i2);
    mu_est->data[0] = ex;

    /* 'gaussian_mixture_simple:32' sig_est=std(x); */
    ex = b_std(x);
    i2 = sig_est->size[0];
    sig_est->size[0] = 1;
    emxEnsureCapacity_real_T(sig_est, i2);
    sig_est->data[0] = ex;

    /* 'gaussian_mixture_simple:33' pp_est=1; */
    i2 = pp_est->size[0];
    pp_est->size[0] = 1;
    emxEnsureCapacity_real_T(pp_est, i2);
    pp_est->data[0] = 1.0;

    /* 'gaussian_mixture_simple:34' l_lik=NaN; */
    *l_lik = rtNaN;
  } else {
    emxInit_real_T(&y_out, 2);
    emxInit_real_T(&edges, 2);

    /* 'gaussian_mixture_simple:35' else */
    /* 'gaussian_mixture_simple:36' Nb=floor(sqrt(length(x))); */
    ex = x->size[0];
    b_sqrt(&ex);
    Nb = floor(ex);

    /* 'gaussian_mixture_simple:37' [y_out,edges]=histcounts(x,Nb); */
    histcounts(x, Nb, y_out, edges);

    /* 'gaussian_mixture_simple:38' mz_out = mean([edges(1:end-1); edges(2:end)], 1); */
    if (1 > edges->size[1] - 1) {
      loop_ub = 0;
    } else {
      loop_ub = edges->size[1] - 1;
    }

    if (2 > edges->size[1]) {
      i2 = 1;
      i3 = 0;
    } else {
      i2 = 2;
      i3 = edges->size[1];
    }

    emxInit_real_T(&b_edges, 2);
    i4 = b_edges->size[0] * b_edges->size[1];
    b_edges->size[0] = 2;
    b_edges->size[1] = loop_ub;
    emxEnsureCapacity_real_T(b_edges, i4);
    for (i4 = 0; i4 < loop_ub; i4++) {
      b_edges->data[i4 << 1] = edges->data[i4];
    }

    loop_ub = i3 - i2;
    for (i3 = 0; i3 <= loop_ub; i3++) {
      b_edges->data[1 + (i3 << 1)] = edges->data[(i2 + i3) - 1];
    }

    emxFree_real_T(&edges);
    emxInit_real_T(&mz_out, 2);
    emxInit_real_T(&aux_mx, 2);
    emxInit_real_T(&p_aux, 2);
    mean(b_edges, mz_out);

    /* 'gaussian_mixture_simple:40' aux_mx=dyn_pr_split_w_aux(mz_out,y_out); */
    dyn_pr_split_w_aux(mz_out, y_out, aux_mx);

    /* 'gaussian_mixture_simple:42' [Q,opt_part]=dyn_pr_split_w(mz_out,y_out,KS-1,aux_mx); */
    /*  initialize */
    /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
    /*  */
    /*         main dynamic programming algorithm */
    /*  */
    /* 'dyn_pr_split_w:9' Q=zeros(1,K_gr); */
    /* 'dyn_pr_split_w:10' N=length(data); */
    varargin_1 = mz_out->size[1];
    N = mz_out->size[1];

    /* 'dyn_pr_split_w:11' p_opt_idx=zeros(1,N); */
    /* 'dyn_pr_split_w:12' p_aux=zeros(1,N); */
    i2 = p_aux->size[0] * p_aux->size[1];
    p_aux->size[0] = 1;
    p_aux->size[1] = mz_out->size[1];
    emxEnsureCapacity_real_T(p_aux, i2);
    loop_ub = mz_out->size[1];
    emxFree_real_T(&b_edges);
    for (i2 = 0; i2 < loop_ub; i2++) {
      p_aux->data[i2] = 0.0;
    }

    emxInit_uint32_T(&opt_pals, 2);

    /* 'dyn_pr_split_w:13' opt_pals=zeros(K_gr,N); */
    i2 = opt_pals->size[0] * opt_pals->size[1];
    loop_ub = (int)(KS - 1.0);
    opt_pals->size[0] = loop_ub;
    opt_pals->size[1] = mz_out->size[1];
    emxEnsureCapacity_uint32_T(opt_pals, i2);
    b_loop_ub = loop_ub * mz_out->size[1];
    for (i2 = 0; i2 < b_loop_ub; i2++) {
      opt_pals->data[i2] = 1U;
    }

    emxInit_real_T(&p_opt_idx, 2);

    /* 'dyn_pr_split_w:14' for kk=1:N */
    i2 = mz_out->size[1];
    i3 = p_opt_idx->size[0] * p_opt_idx->size[1];
    p_opt_idx->size[0] = 1;
    p_opt_idx->size[1] = mz_out->size[1];
    emxEnsureCapacity_real_T(p_opt_idx, i3);
    emxInit_real_T(&b_mz_out, 2);
    emxInit_real_T(&b_y_out, 2);
    for (kk = 0; kk < i2; kk++) {
      /* 'dyn_pr_split_w:15' p_opt_idx(kk)=my_qu_ix_w(data(kk:N),ygreki(kk:N)); */
      if (1 + kk > N) {
        i3 = 0;
        i4 = 0;
        i5 = 0;
        kster = 0;
      } else {
        i3 = kk;
        i4 = varargin_1;
        i5 = kk;
        kster = varargin_1;
      }

      n_tmp_tmp = b_mz_out->size[0] * b_mz_out->size[1];
      b_mz_out->size[0] = 1;
      b_loop_ub = i4 - i3;
      b_mz_out->size[1] = b_loop_ub;
      emxEnsureCapacity_real_T(b_mz_out, n_tmp_tmp);
      for (i4 = 0; i4 < b_loop_ub; i4++) {
        b_mz_out->data[i4] = mz_out->data[i3 + i4];
      }

      i3 = b_y_out->size[0] * b_y_out->size[1];
      b_y_out->size[0] = 1;
      b_loop_ub = kster - i5;
      b_y_out->size[1] = b_loop_ub;
      emxEnsureCapacity_real_T(b_y_out, i3);
      for (i3 = 0; i3 < b_loop_ub; i3++) {
        b_y_out->data[i3] = y_out->data[i5 + i3];
      }

      p_opt_idx->data[kk] = my_qu_ix_w(b_mz_out, b_y_out);
    }

    emxInit_real_T(&Q, 2);

    /*  aux_mx - already computed */
    /*  iterate */
    /* 'dyn_pr_split_w:21' for kster=1:K_gr */
    emxInit_real_T(&c_x, 2);
    for (kster = 0; kster < loop_ub; kster++) {
      /* 'dyn_pr_split_w:22' for kk=1:N-kster */
      i2 = N - kster;
      for (kk = 0; kk <= i2 - 2; kk++) {
        /* 'dyn_pr_split_w:23' for jj=kk+1:N-kster+1 */
        i3 = i2 - kk;
        for (b_loop_ub = 0; b_loop_ub <= i3 - 2; b_loop_ub++) {
          /* 'dyn_pr_split_w:24' p_aux(jj)= aux_mx(kk,jj)+p_opt_idx(jj); */
          i4 = (int)(((1.0 + (double)kk) + 1.0) + (double)b_loop_ub) - 1;
          p_aux->data[i4] = aux_mx->data[kk + aux_mx->size[0] * i4] +
            p_opt_idx->data[i4];
        }

        /* 'dyn_pr_split_w:26' [mm,ix]=min(p_aux(kk+1:N-kster+1)); */
        i3 = (varargin_1 - kster) - 1;
        if ((1.0 + (double)kk) + 1.0 > i3 + 1) {
          i4 = 0;
          i3 = -1;
        } else {
          i4 = kk + 1;
        }

        b_loop_ub = i3 - i4;
        n_tmp_tmp = b_loop_ub + 1;
        n_tmp = (unsigned short)n_tmp_tmp;
        if (n_tmp <= 2) {
          if (n_tmp == 1) {
            p_opt_idx->data[kk] = p_aux->data[i4];
            n_tmp_tmp = 1;
          } else if ((p_aux->data[i4] > p_aux->data[i4 + 1]) || (rtIsNaN
                      (p_aux->data[i4]) && (!rtIsNaN(p_aux->data[i4 + 1])))) {
            p_opt_idx->data[kk] = p_aux->data[i4 + 1];
            n_tmp_tmp = 2;
          } else {
            p_opt_idx->data[kk] = p_aux->data[i4];
            n_tmp_tmp = 1;
          }
        } else {
          i3 = c_x->size[0] * c_x->size[1];
          c_x->size[0] = 1;
          c_x->size[1] = n_tmp_tmp;
          emxEnsureCapacity_real_T(c_x, i3);
          for (i3 = 0; i3 <= b_loop_ub; i3++) {
            c_x->data[i3] = p_aux->data[i4 + i3];
          }

          if (!rtIsNaN(p_aux->data[i4])) {
            n_tmp_tmp = 1;
          } else {
            n_tmp_tmp = 0;
            k = 2;
            exitg1 = false;
            while ((!exitg1) && (k <= n_tmp)) {
              if (!rtIsNaN(c_x->data[k - 1])) {
                n_tmp_tmp = k;
                exitg1 = true;
              } else {
                k++;
              }
            }
          }

          if (n_tmp_tmp == 0) {
            p_opt_idx->data[kk] = p_aux->data[i4];
            n_tmp_tmp = 1;
          } else {
            b_ex = p_aux->data[(i4 + n_tmp_tmp) - 1];
            i3 = n_tmp_tmp + 1;
            for (k = i3; k <= n_tmp; k++) {
              i5 = (i4 + k) - 1;
              if (b_ex > p_aux->data[i5]) {
                b_ex = p_aux->data[i5];
                n_tmp_tmp = k;
              }
            }

            p_opt_idx->data[kk] = b_ex;
          }
        }

        /* 'dyn_pr_split_w:27' p_opt_idx(kk)=mm; */
        /* 'dyn_pr_split_w:28' opt_pals(kster,kk)=kk+ix(1); */
        opt_pals->data[kster + opt_pals->size[0] * kk] = ((unsigned int)kk +
          n_tmp_tmp) + 1U;
      }

      /* 'dyn_pr_split_w:30' Q(kster)=p_opt_idx(1); */
    }

    emxFree_real_T(&p_aux);
    emxFree_real_T(&p_opt_idx);
    emxFree_real_T(&aux_mx);

    /*  restore optimal decisions */
    /* 'dyn_pr_split_w:35' opt_part=zeros(1,K_gr); */
    i2 = Q->size[0] * Q->size[1];
    Q->size[0] = 1;
    Q->size[1] = loop_ub;
    emxEnsureCapacity_real_T(Q, i2);
    for (i2 = 0; i2 < loop_ub; i2++) {
      Q->data[i2] = 0.0;
    }

    /* 'dyn_pr_split_w:36' opt_part(1)=opt_pals(K_gr,1); */
    Q->data[0] = opt_pals->data[loop_ub - 1];

    /* 'dyn_pr_split_w:37' for kster=K_gr-1:-1:1 */
    i2 = (int)((1.0 + (-1.0 - ((KS - 1.0) - 1.0))) / -1.0);
    for (kster = 0; kster < i2; kster++) {
      b_ex = ((KS - 1.0) - 1.0) + -(double)kster;

      /*  	if K_gr-kster+1<=0 */
      /*  		fprintf('K_gr-kster+1<=0\n'); */
      /*  	elseif opt_part(K_gr-kster)<=0 */
      /*  		fprintf('opt_part(K_gr-kster)<=0\n'); */
      /*  	end */
      /* 'dyn_pr_split_w:43' opt_part(K_gr-kster+1)=opt_pals(kster,opt_part(K_gr-kster)); */
      ex = (KS - 1.0) - b_ex;
      Q->data[(int)(ex + 1.0) - 1] = opt_pals->data[((int)b_ex + opt_pals->size
        [0] * ((int)(unsigned int)Q->data[(int)ex - 1] - 1)) - 1];
    }

    emxFree_uint32_T(&opt_pals);
    emxInit_real_T(&part_cl, 2);

    /* 'gaussian_mixture_simple:43' part_cl=[1 opt_part Nb+1]; */
    i2 = part_cl->size[0] * part_cl->size[1];
    part_cl->size[0] = 1;
    part_cl->size[1] = 2 + Q->size[1];
    emxEnsureCapacity_real_T(part_cl, i2);
    part_cl->data[0] = 1.0;
    loop_ub = Q->size[1];
    for (i2 = 0; i2 < loop_ub; i2++) {
      part_cl->data[i2 + 1] = Q->data[i2];
    }

    emxInit_real_T(&pp_ini, 2);
    emxInit_real_T(&mu_ini, 2);
    emxInit_real_T(&sig_ini, 2);
    part_cl->data[1 + Q->size[1]] = Nb + 1.0;

    /*  set initial cond */
    /* 'gaussian_mixture_simple:46' pp_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:47' mu_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:48' sig_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:49' for kkps=1:KS */
    i2 = (int)KS;
    i3 = pp_ini->size[0] * pp_ini->size[1];
    pp_ini->size[0] = 1;
    pp_ini->size[1] = i2;
    emxEnsureCapacity_real_T(pp_ini, i3);
    i3 = mu_ini->size[0] * mu_ini->size[1];
    mu_ini->size[0] = 1;
    mu_ini->size[1] = i2;
    emxEnsureCapacity_real_T(mu_ini, i3);
    i3 = sig_ini->size[0] * sig_ini->size[1];
    sig_ini->size[0] = 1;
    sig_ini->size[1] = i2;
    emxEnsureCapacity_real_T(sig_ini, i3);
    for (b_loop_ub = 0; b_loop_ub < i2; b_loop_ub++) {
      /* 'gaussian_mixture_simple:50' invec=mz_out(part_cl(kkps):part_cl(kkps+1)-1); */
      if (part_cl->data[b_loop_ub] > part_cl->data[1 + b_loop_ub] - 1.0) {
        i3 = 0;
        i4 = -1;
      } else {
        i3 = (int)part_cl->data[b_loop_ub] - 1;
        i4 = (int)(part_cl->data[1 + b_loop_ub] - 1.0) - 1;
      }

      /* 'gaussian_mixture_simple:51' yinwec=y_out(part_cl(kkps):part_cl(kkps+1)-1); */
      if (part_cl->data[b_loop_ub] > part_cl->data[1 + b_loop_ub] - 1.0) {
        i5 = 0;
        kster = 0;
      } else {
        i5 = (int)part_cl->data[b_loop_ub] - 1;
        kster = (int)(part_cl->data[1 + b_loop_ub] - 1.0);
      }

      /* 'gaussian_mixture_simple:52' wwec=yinwec/(sum(yinwec)); */
      /* 'gaussian_mixture_simple:53' pp_ini(kkps)=sum(yinwec)/sum(y_out); */
      n_tmp_tmp = b_y_out->size[0] * b_y_out->size[1];
      b_y_out->size[0] = 1;
      loop_ub = kster - i5;
      b_y_out->size[1] = loop_ub;
      emxEnsureCapacity_real_T(b_y_out, n_tmp_tmp);
      for (kster = 0; kster < loop_ub; kster++) {
        b_y_out->data[kster] = y_out->data[i5 + kster];
      }

      pp_ini->data[b_loop_ub] = b_sum(b_y_out) / b_sum(y_out);

      /* 'gaussian_mixture_simple:54' mu_ini(kkps)=sum(invec.*wwec); */
      kster = b_y_out->size[0] * b_y_out->size[1];
      b_y_out->size[0] = 1;
      b_y_out->size[1] = loop_ub;
      emxEnsureCapacity_real_T(b_y_out, kster);
      for (kster = 0; kster < loop_ub; kster++) {
        b_y_out->data[kster] = y_out->data[i5 + kster];
      }

      ex = b_sum(b_y_out);
      kster = b_mz_out->size[0] * b_mz_out->size[1];
      b_mz_out->size[0] = 1;
      loop_ub = i4 - i3;
      i4 = loop_ub + 1;
      b_mz_out->size[1] = i4;
      emxEnsureCapacity_real_T(b_mz_out, kster);
      for (kster = 0; kster <= loop_ub; kster++) {
        b_mz_out->data[kster] = mz_out->data[i3 + kster] * (y_out->data[i5 +
          kster] / ex);
      }

      mu_ini->data[b_loop_ub] = b_sum(b_mz_out);

      /* sig_ini(kkps)=sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec')); */
      /* 'gaussian_mixture_simple:56' sig_ini(kkps)=0.5*(max(invec)-min(invec)); */
      n_tmp = (unsigned short)i4;
      if (n_tmp <= 2) {
        if (n_tmp == 1) {
          b_ex = mz_out->data[i3];
        } else if ((mz_out->data[i3] < mz_out->data[i3 + 1]) || (rtIsNaN
                    (mz_out->data[i3]) && (!rtIsNaN(mz_out->data[i3 + 1])))) {
          b_ex = mz_out->data[i3 + 1];
        } else {
          b_ex = mz_out->data[i3];
        }
      } else {
        i5 = c_x->size[0] * c_x->size[1];
        c_x->size[0] = 1;
        c_x->size[1] = i4;
        emxEnsureCapacity_real_T(c_x, i5);
        for (i5 = 0; i5 <= loop_ub; i5++) {
          c_x->data[i5] = mz_out->data[i3 + i5];
        }

        if (!rtIsNaN(mz_out->data[i3])) {
          n_tmp_tmp = 1;
        } else {
          n_tmp_tmp = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= n_tmp)) {
            if (!rtIsNaN(c_x->data[k - 1])) {
              n_tmp_tmp = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (n_tmp_tmp == 0) {
          b_ex = mz_out->data[i3];
        } else {
          b_ex = mz_out->data[(i3 + n_tmp_tmp) - 1];
          i5 = n_tmp_tmp + 1;
          for (k = i5; k <= n_tmp; k++) {
            kster = (i3 + k) - 1;
            if (b_ex < mz_out->data[kster]) {
              b_ex = mz_out->data[kster];
            }
          }
        }
      }

      i5 = Q->size[0] * Q->size[1];
      Q->size[0] = 1;
      Q->size[1] = i4;
      emxEnsureCapacity_real_T(Q, i5);
      for (i4 = 0; i4 <= loop_ub; i4++) {
        Q->data[i4] = mz_out->data[i3 + i4];
      }

      if (n_tmp <= 2) {
        if (n_tmp == 1) {
          ex = mz_out->data[i3];
        } else if ((mz_out->data[i3] > mz_out->data[i3 + 1]) || (rtIsNaN
                    (mz_out->data[i3]) && (!rtIsNaN(mz_out->data[i3 + 1])))) {
          ex = mz_out->data[i3 + 1];
        } else {
          ex = mz_out->data[i3];
        }
      } else {
        if (!rtIsNaN(mz_out->data[i3])) {
          n_tmp_tmp = 1;
        } else {
          n_tmp_tmp = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= n_tmp)) {
            if (!rtIsNaN(Q->data[k - 1])) {
              n_tmp_tmp = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (n_tmp_tmp == 0) {
          ex = mz_out->data[i3];
        } else {
          ex = mz_out->data[(i3 + n_tmp_tmp) - 1];
          i4 = n_tmp_tmp + 1;
          for (k = i4; k <= n_tmp; k++) {
            i5 = (i3 + k) - 1;
            if (ex > mz_out->data[i5]) {
              ex = mz_out->data[i5];
            }
          }
        }
      }

      sig_ini->data[b_loop_ub] = 0.5 * (b_ex - ex);
    }

    emxFree_real_T(&b_y_out);
    emxFree_real_T(&b_mz_out);
    emxFree_real_T(&c_x);
    emxFree_real_T(&Q);
    emxFree_real_T(&y_out);
    emxFree_real_T(&part_cl);
    emxFree_real_T(&mz_out);

    /*  */
    /* 'gaussian_mixture_simple:59' [pp_est,mu_est,sig_est,l_lik] = g_mix_est_fast_lik(x,KS,mu_ini,sig_ini,pp_ini); */
    g_mix_est_fast_lik(x, KS, mu_ini, sig_ini, pp_ini, pp_est, mu_est, sig_est,
                       l_lik);
    emxFree_real_T(&sig_ini);
    emxFree_real_T(&mu_ini);
    emxFree_real_T(&pp_ini);
  }

  *TIC = b_TIC;
}

/*
 * File trailer for gaussian_mixture_simple.c
 *
 * [EOF]
 */
