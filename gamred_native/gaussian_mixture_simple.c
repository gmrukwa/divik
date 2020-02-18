/*
 *
 * gaussian_mixture_simple.c
 *
 * Code generation for function 'gaussian_mixture_simple'
 *
 */

/* Include files */
#include "gaussian_mixture_simple.h"
#include "dyn_pr_split_w.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "g_mix_est_fast_lik.h"
#include "histcounts.h"
#include "mean.h"
#include "my_qu_ix_w.h"
#include "rt_nonfinite.h"
#include "std.h"
#include <math.h>

/* Function Definitions */

/*
 * function [pp_est,mu_est,sig_est,TIC,l_lik]=gaussian_mixture_simple(x,counts,KS)
 */
void gaussian_mixture_simple(const emxArray_real_T *x, const emxArray_real_T
  *counts, double KS, emxArray_real_T *pp_est, emxArray_real_T *mu_est,
  emxArray_real_T *sig_est, double *TIC, double *l_lik)
{
  emxArray_real_T *b_x;
  int i;
  int loop_ub;
  int vlen;
  int k;
  emxArray_real_T *y_out;
  emxArray_real_T *edges;
  double y;
  int idx;
  int i1;
  emxArray_real_T *b_edges;
  int i2;
  emxArray_real_T *mz_out;
  emxArray_real_T *aux_mx;
  emxArray_real_T *b_mz_out;
  emxArray_real_T *b_y_out;
  emxArray_real_T *part_cl;
  emxArray_real_T *pp_ini;
  emxArray_real_T *Q;
  int jj;
  int i3;
  int i4;
  int i5;
  int i6;
  emxArray_real_T *mu_ini;
  emxArray_real_T *sig_ini;
  double d;
  double wwec_tmp;
  bool exitg1;
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
  i = b_x->size[0];
  b_x->size[0] = x->size[0];
  emxEnsureCapacity_real_T(b_x, i);
  loop_ub = x->size[0];
  for (i = 0; i < loop_ub; i++) {
    b_x->data[i] = x->data[i] * counts->data[i];
  }

  vlen = b_x->size[0];
  if (b_x->size[0] == 0) {
    *TIC = 0.0;
  } else {
    *TIC = b_x->data[0];
    for (k = 2; k <= vlen; k++) {
      *TIC += b_x->data[k - 1];
    }
  }

  emxFree_real_T(&b_x);

  /* 'gaussian_mixture_simple:30' if KS==1 */
  if (KS == 1.0) {
    /* 'gaussian_mixture_simple:31' mu_est=nanmean(x); */
    if (x->size[0] == 0) {
      y = rtNaN;
    } else {
      vlen = x->size[0];
      y = 0.0;
      idx = 0;
      for (k = 0; k < vlen; k++) {
        if (!rtIsNaN(x->data[k])) {
          y += x->data[k];
          idx++;
        }
      }

      if (idx == 0) {
        y = rtNaN;
      } else {
        y /= (double)idx;
      }
    }

    i = mu_est->size[0];
    mu_est->size[0] = 1;
    emxEnsureCapacity_real_T(mu_est, i);
    mu_est->data[0] = y;

    /* 'gaussian_mixture_simple:32' sig_est=std(x); */
    i = sig_est->size[0];
    sig_est->size[0] = 1;
    emxEnsureCapacity_real_T(sig_est, i);
    sig_est->data[0] = b_std(x);

    /* 'gaussian_mixture_simple:33' pp_est=1; */
    i = pp_est->size[0];
    pp_est->size[0] = 1;
    emxEnsureCapacity_real_T(pp_est, i);
    pp_est->data[0] = 1.0;

    /* 'gaussian_mixture_simple:34' l_lik=NaN; */
    *l_lik = rtNaN;
  } else {
    emxInit_real_T(&y_out, 2);
    emxInit_real_T(&edges, 2);

    /* 'gaussian_mixture_simple:35' else */
    /* 'gaussian_mixture_simple:36' Nb=floor(sqrt(length(x))); */
    vlen = (int)floor(sqrt(x->size[0]));

    /* 'gaussian_mixture_simple:37' [y_out,edges]=histcounts(x,Nb); */
    histcounts(x, vlen, y_out, edges);

    /* 'gaussian_mixture_simple:38' mz_out = mean([edges(1:end-1); edges(2:end)], 1); */
    if (1 > edges->size[1] - 1) {
      loop_ub = 0;
    } else {
      loop_ub = edges->size[1] - 1;
    }

    if (2 > edges->size[1]) {
      i = 0;
      i1 = 0;
    } else {
      i = 1;
      i1 = edges->size[1];
    }

    emxInit_real_T(&b_edges, 2);
    i2 = b_edges->size[0] * b_edges->size[1];
    b_edges->size[0] = 2;
    b_edges->size[1] = loop_ub;
    emxEnsureCapacity_real_T(b_edges, i2);
    for (i2 = 0; i2 < loop_ub; i2++) {
      b_edges->data[2 * i2] = edges->data[i2];
    }

    loop_ub = i1 - i;
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_edges->data[2 * i1 + 1] = edges->data[i + i1];
    }

    emxFree_real_T(&edges);
    emxInit_real_T(&mz_out, 2);
    emxInit_real_T(&aux_mx, 2);
    mean(b_edges, mz_out);

    /* 'gaussian_mixture_simple:40' aux_mx=dyn_pr_split_w_aux(mz_out,y_out); */
    /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
    /*  */
    /*  */
    /*          auxiliary - compute quality index matrix */
    /*  */
    /* 'dyn_pr_split_w_aux:8' N=length(data); */
    /*  aux_mx */
    /* 'dyn_pr_split_w_aux:11' aux_mx=zeros(N,N); */
    i = aux_mx->size[0] * aux_mx->size[1];
    aux_mx->size[0] = mz_out->size[1];
    aux_mx->size[1] = mz_out->size[1];
    emxEnsureCapacity_real_T(aux_mx, i);
    loop_ub = mz_out->size[1] * mz_out->size[1];
    emxFree_real_T(&b_edges);
    for (i = 0; i < loop_ub; i++) {
      aux_mx->data[i] = 0.0;
    }

    /* 'dyn_pr_split_w_aux:12' for kk=1:N-1 */
    i = mz_out->size[1];
    emxInit_real_T(&b_mz_out, 2);
    emxInit_real_T(&b_y_out, 2);
    for (idx = 0; idx <= i - 2; idx++) {
      /* 'dyn_pr_split_w_aux:13' for jj=kk+1:N */
      i1 = mz_out->size[1] - idx;
      for (k = 0; k <= i1 - 2; k++) {
        jj = idx + k;

        /* 'dyn_pr_split_w_aux:14' aux_mx(kk,jj)= my_qu_ix_w(data(kk:jj-1),ygreki(kk:jj-1)); */
        if (idx + 1 > jj + 1) {
          i2 = 0;
          i3 = -1;
          i4 = 0;
          i5 = -1;
        } else {
          i2 = idx;
          i3 = jj;
          i4 = idx;
          i5 = jj;
        }

        i6 = b_mz_out->size[0] * b_mz_out->size[1];
        b_mz_out->size[0] = 1;
        loop_ub = i3 - i2;
        b_mz_out->size[1] = loop_ub + 1;
        emxEnsureCapacity_real_T(b_mz_out, i6);
        for (i3 = 0; i3 <= loop_ub; i3++) {
          b_mz_out->data[i3] = mz_out->data[i2 + i3];
        }

        i2 = b_y_out->size[0] * b_y_out->size[1];
        b_y_out->size[0] = 1;
        loop_ub = i5 - i4;
        b_y_out->size[1] = loop_ub + 1;
        emxEnsureCapacity_real_T(b_y_out, i2);
        for (i2 = 0; i2 <= loop_ub; i2++) {
          b_y_out->data[i2] = y_out->data[i4 + i2];
        }

        aux_mx->data[idx + aux_mx->size[0] * (jj + 1)] = my_qu_ix_w(b_mz_out,
          b_y_out);
      }
    }

    emxFree_real_T(&b_y_out);
    emxFree_real_T(&b_mz_out);
    emxInit_real_T(&part_cl, 2);
    emxInit_real_T(&pp_ini, 2);
    emxInit_real_T(&Q, 2);

    /* 'gaussian_mixture_simple:42' [Q,opt_part]=dyn_pr_split_w(mz_out,y_out,KS-1,aux_mx); */
    dyn_pr_split_w(mz_out, y_out, KS - 1.0, aux_mx, Q, pp_ini);

    /* 'gaussian_mixture_simple:43' part_cl=[1 opt_part Nb+1]; */
    i = part_cl->size[0] * part_cl->size[1];
    part_cl->size[0] = 1;
    part_cl->size[1] = pp_ini->size[1] + 2;
    emxEnsureCapacity_real_T(part_cl, i);
    part_cl->data[0] = 1.0;
    loop_ub = pp_ini->size[1];
    emxFree_real_T(&aux_mx);
    for (i = 0; i < loop_ub; i++) {
      part_cl->data[i + 1] = pp_ini->data[i];
    }

    emxInit_real_T(&mu_ini, 2);
    emxInit_real_T(&sig_ini, 2);
    part_cl->data[pp_ini->size[1] + 1] = (double)vlen + 1.0;

    /*  set initial cond */
    /* 'gaussian_mixture_simple:46' pp_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:47' mu_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:48' sig_ini=zeros(1,KS); */
    /* 'gaussian_mixture_simple:49' for kkps=1:KS */
    i = (int)KS;
    i1 = pp_ini->size[0] * pp_ini->size[1];
    pp_ini->size[0] = 1;
    pp_ini->size[1] = i;
    emxEnsureCapacity_real_T(pp_ini, i1);
    i1 = mu_ini->size[0] * mu_ini->size[1];
    mu_ini->size[0] = 1;
    mu_ini->size[1] = i;
    emxEnsureCapacity_real_T(mu_ini, i1);
    i1 = sig_ini->size[0] * sig_ini->size[1];
    sig_ini->size[0] = 1;
    sig_ini->size[1] = i;
    emxEnsureCapacity_real_T(sig_ini, i1);
    for (jj = 0; jj < i; jj++) {
      /* 'gaussian_mixture_simple:50' invec=mz_out(part_cl(kkps):part_cl(kkps+1)-1); */
      d = part_cl->data[jj];
      wwec_tmp = part_cl->data[jj + 1];
      if (d > wwec_tmp - 1.0) {
        i1 = 0;
        i2 = -1;
      } else {
        i1 = (int)d - 1;
        i2 = (int)(wwec_tmp - 1.0) - 1;
      }

      /* 'gaussian_mixture_simple:51' yinwec=y_out(part_cl(kkps):part_cl(kkps+1)-1); */
      if (d > wwec_tmp - 1.0) {
        i3 = -1;
        i4 = -1;
      } else {
        i3 = (int)d - 2;
        i4 = (int)(wwec_tmp - 1.0) - 1;
      }

      /* 'gaussian_mixture_simple:52' wwec=yinwec/(sum(yinwec)); */
      i5 = Q->size[0] * Q->size[1];
      Q->size[0] = 1;
      loop_ub = i4 - i3;
      Q->size[1] = loop_ub;
      emxEnsureCapacity_real_T(Q, i5);
      for (i4 = 0; i4 < loop_ub; i4++) {
        Q->data[i4] = y_out->data[(i3 + i4) + 1];
      }

      if (loop_ub == 0) {
        wwec_tmp = 0.0;
      } else {
        wwec_tmp = y_out->data[i3 + 1];
        for (k = 2; k <= loop_ub; k++) {
          wwec_tmp += Q->data[k - 1];
        }
      }

      /* 'gaussian_mixture_simple:53' pp_ini(kkps)=sum(yinwec)/sum(y_out); */
      vlen = y_out->size[1];
      if (y_out->size[1] == 0) {
        y = 0.0;
      } else {
        y = y_out->data[0];
        for (k = 2; k <= vlen; k++) {
          y += y_out->data[k - 1];
        }
      }

      pp_ini->data[jj] = wwec_tmp / y;

      /* 'gaussian_mixture_simple:54' mu_ini(kkps)=sum(invec.*wwec); */
      i4 = Q->size[0] * Q->size[1];
      Q->size[0] = 1;
      loop_ub = i2 - i1;
      i2 = loop_ub + 1;
      Q->size[1] = i2;
      emxEnsureCapacity_real_T(Q, i4);
      for (i4 = 0; i4 <= loop_ub; i4++) {
        Q->data[i4] = mz_out->data[i1 + i4] * (y_out->data[(i3 + i4) + 1] /
          wwec_tmp);
      }

      i3 = Q->size[1];
      if (Q->size[1] == 0) {
        y = 0.0;
      } else {
        y = Q->data[0];
        for (k = 2; k <= i3; k++) {
          y += Q->data[k - 1];
        }
      }

      mu_ini->data[jj] = y;

      /* sig_ini(kkps)=sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec')); */
      /* 'gaussian_mixture_simple:56' sig_ini(kkps)=0.5*(max(invec)-min(invec)); */
      vlen = (unsigned short)i2;
      if (vlen <= 2) {
        if (vlen == 1) {
          wwec_tmp = mz_out->data[i1];
        } else {
          wwec_tmp = mz_out->data[i1 + 1];
          if ((!(mz_out->data[i1] < wwec_tmp)) && ((!rtIsNaN(mz_out->data[i1])) ||
               rtIsNaN(wwec_tmp))) {
            wwec_tmp = mz_out->data[i1];
          }
        }
      } else {
        i3 = Q->size[0] * Q->size[1];
        Q->size[0] = 1;
        Q->size[1] = i2;
        emxEnsureCapacity_real_T(Q, i3);
        for (i3 = 0; i3 <= loop_ub; i3++) {
          Q->data[i3] = mz_out->data[i1 + i3];
        }

        if (!rtIsNaN(mz_out->data[i1])) {
          idx = 1;
        } else {
          idx = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= vlen)) {
            if (!rtIsNaN(Q->data[k - 1])) {
              idx = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (idx == 0) {
          wwec_tmp = mz_out->data[i1];
        } else {
          wwec_tmp = mz_out->data[(i1 + idx) - 1];
          i3 = idx + 1;
          for (k = i3; k <= vlen; k++) {
            d = mz_out->data[(i1 + k) - 1];
            if (wwec_tmp < d) {
              wwec_tmp = d;
            }
          }
        }
      }

      i3 = Q->size[0] * Q->size[1];
      Q->size[0] = 1;
      Q->size[1] = i2;
      emxEnsureCapacity_real_T(Q, i3);
      for (i2 = 0; i2 <= loop_ub; i2++) {
        Q->data[i2] = mz_out->data[i1 + i2];
      }

      if (vlen <= 2) {
        if (vlen == 1) {
          y = mz_out->data[i1];
        } else {
          y = mz_out->data[i1 + 1];
          if ((!(mz_out->data[i1] > y)) && ((!rtIsNaN(mz_out->data[i1])) ||
               rtIsNaN(mz_out->data[i1 + 1]))) {
            y = mz_out->data[i1];
          }
        }
      } else {
        if (!rtIsNaN(mz_out->data[i1])) {
          idx = 1;
        } else {
          idx = 0;
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= vlen)) {
            if (!rtIsNaN(Q->data[k - 1])) {
              idx = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }

        if (idx == 0) {
          y = mz_out->data[i1];
        } else {
          y = mz_out->data[(i1 + idx) - 1];
          i2 = idx + 1;
          for (k = i2; k <= vlen; k++) {
            d = mz_out->data[(i1 + k) - 1];
            if (y > d) {
              y = d;
            }
          }
        }
      }

      sig_ini->data[jj] = 0.5 * (wwec_tmp - y);
    }

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
}

/* End of code generation (gaussian_mixture_simple.c) */
