/*
 *
 * g_mix_est_fast_lik.c
 *
 * Code generation for function 'g_mix_est_fast_lik'
 *
 */

/* Include files */
#include "g_mix_est_fast_lik.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "fetch_thresholds_rtwutil.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include "sum.h"
#include <math.h>

/* Function Definitions */

/*
 * function [ppsort,musort,sigsort,l_lik] = g_mix_est_fast_lik(raw_sample,KS,muv,sigv,pp)
 */
void g_mix_est_fast_lik(const emxArray_real_T *raw_sample, double KS, const
  emxArray_real_T *muv, const emxArray_real_T *sigv, const emxArray_real_T *pp,
  emxArray_real_T *ppsort, emxArray_real_T *musort, emxArray_real_T *sigsort,
  double *l_lik)
{
  emxArray_real_T *varargin_1;
  int i;
  int nx;
  int k;
  emxArray_real_T *raw_sigvoc;
  int loop_ub;
  emxArray_real_T *sigvoc;
  emxArray_real_T *ppoc;
  emxArray_real_T *pssmac;
  int b_varargin_1;
  int aoffset;
  double change;
  int rescue_iterations_count;
  emxArray_real_T *oldppoc;
  emxArray_real_T *oldsigvoc;
  emxArray_real_T *pskmac;
  emxArray_real_T *ppp;
  emxArray_real_T *sigmac;
  emxArray_real_T *b;
  emxArray_real_T *r;
  emxArray_real_T *b_musort;
  emxArray_real_T *r1;
  emxArray_real_T *c_musort;
  emxArray_int32_T *iidx;
  unsigned int unnamed_idx_1;
  int m;
  double t;
  int inner;
  int i1;
  int b_loop_ub;
  int i2;
  emxInit_real_T(&varargin_1, 2);

  /*  EM iterations  */
  /*  input data: */
  /*  sample - vector of observations */
  /*  muv,sigv,pp - initial values of mixture parameters */
  /*  output: */
  /*  musort,sigsort,ppsort - estimated mixture parameters */
  /*  l_lik - log likeliood */
  /* 'g_mix_est_fast_lik:10' min_sg=0.1; */
  /* 'g_mix_est_fast_lik:12' raw_sigvoc=max(sigv.^2, min_sg^2); */
  i = varargin_1->size[0] * varargin_1->size[1];
  varargin_1->size[0] = 1;
  varargin_1->size[1] = sigv->size[1];
  emxEnsureCapacity_real_T(varargin_1, i);
  nx = sigv->size[1];
  for (k = 0; k < nx; k++) {
    varargin_1->data[k] = rt_powd_snf(sigv->data[k], 2.0);
  }

  emxInit_real_T(&raw_sigvoc, 2);
  i = raw_sigvoc->size[0] * raw_sigvoc->size[1];
  raw_sigvoc->size[0] = 1;
  raw_sigvoc->size[1] = varargin_1->size[1];
  emxEnsureCapacity_real_T(raw_sigvoc, i);
  nx = varargin_1->size[1];
  for (k = 0; k < nx; k++) {
    raw_sigvoc->data[k] = fmax(varargin_1->data[k], 0.010000000000000002);
  }

  /* 'g_mix_est_fast_lik:14' sample = raw_sample(:)'; */
  /* 'g_mix_est_fast_lik:15' mivoc = muv(:); */
  i = musort->size[0];
  musort->size[0] = muv->size[1];
  emxEnsureCapacity_real_T(musort, i);
  loop_ub = muv->size[1];
  for (i = 0; i < loop_ub; i++) {
    musort->data[i] = muv->data[i];
  }

  emxInit_real_T(&sigvoc, 1);

  /* 'g_mix_est_fast_lik:16' sigvoc = raw_sigvoc(:); */
  i = sigvoc->size[0];
  sigvoc->size[0] = raw_sigvoc->size[1];
  emxEnsureCapacity_real_T(sigvoc, i);
  loop_ub = raw_sigvoc->size[1];
  for (i = 0; i < loop_ub; i++) {
    sigvoc->data[i] = raw_sigvoc->data[i];
  }

  emxFree_real_T(&raw_sigvoc);
  emxInit_real_T(&ppoc, 1);

  /* 'g_mix_est_fast_lik:17' ppoc = pp(:); */
  i = ppoc->size[0];
  ppoc->size[0] = pp->size[1];
  emxEnsureCapacity_real_T(ppoc, i);
  loop_ub = pp->size[1];
  for (i = 0; i < loop_ub; i++) {
    ppoc->data[i] = pp->data[i];
  }

  emxInit_real_T(&pssmac, 2);

  /* 'g_mix_est_fast_lik:21' N=length(sample); */
  b_varargin_1 = raw_sample->size[0];

  /*  OK oceniamy iteracyjnie wg wzorow z artykulu Bilmsa */
  /* 'g_mix_est_fast_lik:24' pssmac=zeros(KS,N); */
  loop_ub = (int)KS;
  i = pssmac->size[0] * pssmac->size[1];
  pssmac->size[0] = loop_ub;
  pssmac->size[1] = raw_sample->size[0];
  emxEnsureCapacity_real_T(pssmac, i);
  aoffset = loop_ub * raw_sample->size[0];
  for (i = 0; i < aoffset; i++) {
    pssmac->data[i] = 0.0;
  }

  /* 'g_mix_est_fast_lik:25' change=1; */
  change = 1.0;

  /* 'g_mix_est_fast_lik:26' rescue_iterations_count = 0; */
  rescue_iterations_count = 0;

  /* 'g_mix_est_fast_lik:27' unreachable_number_of_iterations = 10000; */
  /* 'g_mix_est_fast_lik:29' while change > 1.5e-4 && rescue_iterations_count < unreachable_number_of_iterations */
  emxInit_real_T(&oldppoc, 1);
  emxInit_real_T(&oldsigvoc, 1);
  emxInit_real_T(&pskmac, 2);
  emxInit_real_T(&ppp, 1);
  emxInit_real_T(&sigmac, 2);
  emxInit_real_T(&b, 1);
  emxInit_real_T(&r, 2);
  emxInit_real_T(&b_musort, 2);
  emxInit_real_T(&r1, 2);
  emxInit_real_T(&c_musort, 2);
  while ((change > 0.00015) && (rescue_iterations_count < 10000)) {
    /* 'g_mix_est_fast_lik:30' rescue_iterations_count = rescue_iterations_count + 1; */
    rescue_iterations_count++;

    /* 'g_mix_est_fast_lik:31' oldppoc=ppoc; */
    i = oldppoc->size[0];
    oldppoc->size[0] = ppoc->size[0];
    emxEnsureCapacity_real_T(oldppoc, i);
    aoffset = ppoc->size[0];
    for (i = 0; i < aoffset; i++) {
      oldppoc->data[i] = ppoc->data[i];
    }

    /* 'g_mix_est_fast_lik:32' oldsigvoc=sigvoc; */
    i = oldsigvoc->size[0];
    oldsigvoc->size[0] = sigvoc->size[0];
    emxEnsureCapacity_real_T(oldsigvoc, i);
    aoffset = sigvoc->size[0];
    for (i = 0; i < aoffset; i++) {
      oldsigvoc->data[i] = sigvoc->data[i];
    }

    /*  lower limit for component weights */
    /* 'g_mix_est_fast_lik:35' ppoc=max(ppoc,0.001); */
    i = ppp->size[0];
    ppp->size[0] = ppoc->size[0];
    emxEnsureCapacity_real_T(ppp, i);
    aoffset = ppoc->size[0];
    for (i = 0; i < aoffset; i++) {
      ppp->data[i] = ppoc->data[i];
    }

    i = ppoc->size[0];
    for (k = 0; k < i; k++) {
      ppoc->data[k] = fmax(ppp->data[k], 0.001);
    }

    /* 'g_mix_est_fast_lik:37' for kskla=1:KS */
    i = loop_ub - 1;
    if (0 <= i) {
      unnamed_idx_1 = (unsigned int)raw_sample->size[0];
    }

    for (nx = 0; nx < loop_ub; nx++) {
      /* 'g_mix_est_fast_lik:38' pssmac(kskla,:)=ppoc(kskla)*normpdf(sample,mivoc(kskla),sqrt(sigvoc(kskla))); */
      change = sqrt(sigvoc->data[nx]);
      k = varargin_1->size[0] * varargin_1->size[1];
      varargin_1->size[0] = 1;
      m = (int)unnamed_idx_1;
      varargin_1->size[1] = (int)unnamed_idx_1;
      emxEnsureCapacity_real_T(varargin_1, k);
      for (k = 0; k < m; k++) {
        if (change > 0.0) {
          t = (raw_sample->data[k] - musort->data[nx]) / change;
          varargin_1->data[k] = exp(-0.5 * t * t) / (2.5066282746310002 * change);
        } else {
          varargin_1->data[k] = rtNaN;
        }
      }

      aoffset = varargin_1->size[1];
      for (k = 0; k < aoffset; k++) {
        pssmac->data[nx + pssmac->size[0] * k] = ppoc->data[nx] *
          varargin_1->data[k];
      }
    }

    /* 'g_mix_est_fast_lik:40' psummac=ones(KS,1)*sum(pssmac,1); */
    /* 'g_mix_est_fast_lik:41' pskmac=pssmac./psummac; */
    sum(pssmac, varargin_1);
    k = pskmac->size[0] * pskmac->size[1];
    pskmac->size[0] = loop_ub;
    pskmac->size[1] = varargin_1->size[1];
    emxEnsureCapacity_real_T(pskmac, k);
    for (k = 0; k < loop_ub; k++) {
      aoffset = varargin_1->size[1];
      for (m = 0; m < aoffset; m++) {
        pskmac->data[k + pskmac->size[0] * m] = pssmac->data[k + pssmac->size[0]
          * m] / varargin_1->data[m];
      }
    }

    /* 'g_mix_est_fast_lik:42' ppp=sum(pskmac,2); */
    b_sum(pskmac, ppp);

    /* 'g_mix_est_fast_lik:43' ppoc=ppp/N; */
    k = ppoc->size[0];
    ppoc->size[0] = ppp->size[0];
    emxEnsureCapacity_real_T(ppoc, k);
    aoffset = ppp->size[0];
    for (k = 0; k < aoffset; k++) {
      ppoc->data[k] = ppp->data[k] / (double)b_varargin_1;
    }

    /* 'g_mix_est_fast_lik:44' mivoc=pskmac*sample'; */
    if ((pskmac->size[1] == 1) || (raw_sample->size[0] == 1)) {
      k = musort->size[0];
      musort->size[0] = pskmac->size[0];
      emxEnsureCapacity_real_T(musort, k);
      aoffset = pskmac->size[0];
      for (k = 0; k < aoffset; k++) {
        musort->data[k] = 0.0;
        nx = pskmac->size[1];
        for (m = 0; m < nx; m++) {
          musort->data[k] += pskmac->data[k + pskmac->size[0] * m] *
            raw_sample->data[m];
        }
      }
    } else {
      m = pskmac->size[0];
      inner = pskmac->size[1];
      k = musort->size[0];
      musort->size[0] = pskmac->size[0];
      emxEnsureCapacity_real_T(musort, k);
      for (nx = 0; nx < m; nx++) {
        musort->data[nx] = 0.0;
      }

      for (k = 0; k < inner; k++) {
        aoffset = k * m;
        for (nx = 0; nx < m; nx++) {
          musort->data[nx] += raw_sample->data[k] * pskmac->data[aoffset + nx];
        }
      }
    }

    /* 'g_mix_est_fast_lik:45' mivoc=mivoc./ppp; */
    aoffset = musort->size[0];
    for (k = 0; k < aoffset; k++) {
      musort->data[k] /= ppp->data[k];
    }

    /* 'g_mix_est_fast_lik:46' sigmac=(ones(KS,1)*sample-mivoc*ones(1,N)).*((ones(KS,1)*sample-mivoc*ones(1,N))); */
    k = r->size[0] * r->size[1];
    r->size[0] = loop_ub;
    r->size[1] = raw_sample->size[0];
    emxEnsureCapacity_real_T(r, k);
    aoffset = raw_sample->size[0];
    for (k = 0; k < aoffset; k++) {
      for (m = 0; m < loop_ub; m++) {
        r->data[m + r->size[0] * k] = raw_sample->data[k];
      }
    }

    k = b_musort->size[0] * b_musort->size[1];
    b_musort->size[0] = musort->size[0];
    b_musort->size[1] = b_varargin_1;
    emxEnsureCapacity_real_T(b_musort, k);
    for (k = 0; k < b_varargin_1; k++) {
      aoffset = musort->size[0];
      for (m = 0; m < aoffset; m++) {
        b_musort->data[m + b_musort->size[0] * k] = musort->data[m];
      }
    }

    k = r1->size[0] * r1->size[1];
    r1->size[0] = loop_ub;
    r1->size[1] = raw_sample->size[0];
    emxEnsureCapacity_real_T(r1, k);
    aoffset = raw_sample->size[0];
    for (k = 0; k < aoffset; k++) {
      for (m = 0; m < loop_ub; m++) {
        r1->data[m + r1->size[0] * k] = raw_sample->data[k];
      }
    }

    k = c_musort->size[0] * c_musort->size[1];
    c_musort->size[0] = musort->size[0];
    c_musort->size[1] = b_varargin_1;
    emxEnsureCapacity_real_T(c_musort, k);
    for (k = 0; k < b_varargin_1; k++) {
      aoffset = musort->size[0];
      for (m = 0; m < aoffset; m++) {
        c_musort->data[m + c_musort->size[0] * k] = musort->data[m];
      }
    }

    k = sigmac->size[0] * sigmac->size[1];
    sigmac->size[0] = r->size[0];
    sigmac->size[1] = r->size[1];
    emxEnsureCapacity_real_T(sigmac, k);
    aoffset = r->size[0] * r->size[1];
    for (k = 0; k < aoffset; k++) {
      sigmac->data[k] = (r->data[k] - b_musort->data[k]) * (r1->data[k] -
        c_musort->data[k]);
    }

    /* 'g_mix_est_fast_lik:47' for kkk=1:KS */
    if (0 <= i) {
      i1 = sigmac->size[1];
      b_loop_ub = sigmac->size[1];
      i2 = pskmac->size[1];
    }

    for (nx = 0; nx < loop_ub; nx++) {
      /*  lower limit for component variances  */
      /* 'g_mix_est_fast_lik:49' sigvoc(kkk)=max([pskmac(kkk,:)*sigmac(kkk,:)' min_sg^2]); */
      i = b->size[0];
      b->size[0] = i1;
      emxEnsureCapacity_real_T(b, i);
      for (i = 0; i < b_loop_ub; i++) {
        b->data[i] = sigmac->data[nx + sigmac->size[0] * i];
      }

      if ((i2 == 1) || (b->size[0] == 1)) {
        aoffset = pskmac->size[1];
        change = 0.0;
        for (i = 0; i < aoffset; i++) {
          change += pskmac->data[nx + pskmac->size[0] * i] * b->data[i];
        }
      } else {
        aoffset = pskmac->size[1];
        change = 0.0;
        for (i = 0; i < aoffset; i++) {
          change += pskmac->data[nx + pskmac->size[0] * i] * b->data[i];
        }
      }

      if ((change < 0.010000000000000002) || rtIsNaN(change)) {
        sigvoc->data[nx] = 0.010000000000000002;
      } else {
        sigvoc->data[nx] = change;
      }
    }

    /* 'g_mix_est_fast_lik:51' sigvoc=sigvoc./ppp; */
    aoffset = sigvoc->size[0];
    for (i = 0; i < aoffset; i++) {
      sigvoc->data[i] /= ppp->data[i];
    }

    /*  */
    /* 'g_mix_est_fast_lik:54' change=sum(abs(ppoc-oldppoc))+sum(abs(sigvoc-oldsigvoc))/KS; */
    i = oldppoc->size[0];
    oldppoc->size[0] = ppoc->size[0];
    emxEnsureCapacity_real_T(oldppoc, i);
    aoffset = ppoc->size[0];
    for (i = 0; i < aoffset; i++) {
      oldppoc->data[i] = ppoc->data[i] - oldppoc->data[i];
    }

    nx = oldppoc->size[0];
    i = b->size[0];
    b->size[0] = oldppoc->size[0];
    emxEnsureCapacity_real_T(b, i);
    for (k = 0; k < nx; k++) {
      b->data[k] = fabs(oldppoc->data[k]);
    }

    i = oldsigvoc->size[0];
    oldsigvoc->size[0] = sigvoc->size[0];
    emxEnsureCapacity_real_T(oldsigvoc, i);
    aoffset = sigvoc->size[0];
    for (i = 0; i < aoffset; i++) {
      oldsigvoc->data[i] = sigvoc->data[i] - oldsigvoc->data[i];
    }

    nx = oldsigvoc->size[0];
    i = ppp->size[0];
    ppp->size[0] = oldsigvoc->size[0];
    emxEnsureCapacity_real_T(ppp, i);
    for (k = 0; k < nx; k++) {
      ppp->data[k] = fabs(oldsigvoc->data[k]);
    }

    nx = b->size[0];
    if (b->size[0] == 0) {
      change = 0.0;
    } else {
      change = b->data[0];
      for (k = 2; k <= nx; k++) {
        change += b->data[k - 1];
      }
    }

    nx = ppp->size[0];
    if (ppp->size[0] == 0) {
      t = 0.0;
    } else {
      t = ppp->data[0];
      for (k = 2; k <= nx; k++) {
        t += ppp->data[k - 1];
      }
    }

    change += t / KS;
  }

  emxFree_real_T(&c_musort);
  emxFree_real_T(&r1);
  emxFree_real_T(&b_musort);
  emxFree_real_T(&r);
  emxFree_real_T(&b);
  emxFree_real_T(&sigmac);
  emxFree_real_T(&pskmac);
  emxFree_real_T(&oldsigvoc);
  emxFree_real_T(&oldppoc);

  /*  compute likelihood */
  /* 'g_mix_est_fast_lik:58' l_lik=sum(log(sum(pssmac,1))); */
  sum(pssmac, varargin_1);
  nx = varargin_1->size[1];
  emxFree_real_T(&pssmac);
  for (k = 0; k < nx; k++) {
    varargin_1->data[k] = log(varargin_1->data[k]);
  }

  nx = varargin_1->size[1];
  if (varargin_1->size[1] == 0) {
    *l_lik = 0.0;
  } else {
    *l_lik = varargin_1->data[0];
    for (k = 2; k <= nx; k++) {
      *l_lik += varargin_1->data[k - 1];
    }
  }

  emxFree_real_T(&varargin_1);
  emxInit_int32_T(&iidx, 1);

  /*  sort estimates */
  /* 'g_mix_est_fast_lik:61' [musort,isort]=sort(mivoc); */
  sort(musort, iidx);
  i = ppp->size[0];
  ppp->size[0] = iidx->size[0];
  emxEnsureCapacity_real_T(ppp, i);
  loop_ub = iidx->size[0];
  for (i = 0; i < loop_ub; i++) {
    ppp->data[i] = iidx->data[i];
  }

  emxFree_int32_T(&iidx);

  /* 'g_mix_est_fast_lik:62' sigsort=sqrt(sigvoc(isort)); */
  i = sigsort->size[0];
  sigsort->size[0] = ppp->size[0];
  emxEnsureCapacity_real_T(sigsort, i);
  loop_ub = ppp->size[0];
  for (i = 0; i < loop_ub; i++) {
    sigsort->data[i] = sigvoc->data[(int)ppp->data[i] - 1];
  }

  emxFree_real_T(&sigvoc);
  nx = ppp->size[0];
  for (k = 0; k < nx; k++) {
    sigsort->data[k] = sqrt(sigsort->data[k]);
  }

  /* 'g_mix_est_fast_lik:63' ppsort=ppoc(isort); */
  i = ppsort->size[0];
  ppsort->size[0] = ppp->size[0];
  emxEnsureCapacity_real_T(ppsort, i);
  loop_ub = ppp->size[0];
  for (i = 0; i < loop_ub; i++) {
    ppsort->data[i] = ppoc->data[(int)ppp->data[i] - 1];
  }

  emxFree_real_T(&ppp);
  emxFree_real_T(&ppoc);
}

/* End of code generation (g_mix_est_fast_lik.c) */
