/*
 * File: g_mix_est_fast_lik.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:16:52
 */

/* Include Files */
#include <math.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "g_mix_est_fast_lik.h"
#include "sum.h"
#include "abs.h"
#include "fetch_thresholds_emxutil.h"
#include "rdivide_helper.h"
#include "sort1.h"
#include "fetch_thresholds_rtwutil.h"

/* Function Definitions */

/*
 * function [ppsort,musort,sigsort,l_lik] = g_mix_est_fast_lik(raw_sample,KS,muv,sigv,pp)
 * input data:
 *  sample - vector of observations
 *  muv,sigv,pp - initial values of mixture parameters
 *  output:
 *  musort,sigsort,ppsort - estimated mixture parameters
 *  l_lik - log likeliood
 * Arguments    : const emxArray_real_T *raw_sample
 *                double KS
 *                const emxArray_real_T *muv
 *                const emxArray_real_T *sigv
 *                const emxArray_real_T *pp
 *                emxArray_real_T *ppsort
 *                emxArray_real_T *musort
 *                emxArray_real_T *sigsort
 *                double *l_lik
 * Return Type  : void
 */
void g_mix_est_fast_lik(const emxArray_real_T *raw_sample, double KS, const
  emxArray_real_T *muv, const emxArray_real_T *sigv, const emxArray_real_T *pp,
  emxArray_real_T *ppsort, emxArray_real_T *musort, emxArray_real_T *sigsort,
  double *l_lik)
{
  emxArray_real_T *y;
  int i14;
  int nx;
  int k;
  emxArray_real_T *raw_sigvoc;
  double change;
  int loop_ub;
  emxArray_real_T *sigvoc;
  emxArray_real_T *ppoc;
  emxArray_real_T *pssmac;
  unsigned int unnamed_idx_1;
  int varargin_1;
  int b_unnamed_idx_1;
  int vlen;
  int rescue_iterations_count;
  emxArray_real_T *oldppoc;
  emxArray_real_T *oldsigvoc;
  emxArray_real_T *pskmac;
  emxArray_real_T *ppp;
  emxArray_real_T *sigmac;
  emxArray_real_T *b;
  emxArray_real_T *r0;
  emxArray_real_T *b_musort;
  emxArray_real_T *r1;
  emxArray_real_T *c_musort;
  emxArray_real_T *d_musort;
  emxArray_int32_T *iidx;
  double t;
  int vstride;
  int xoffset;
  emxInit_real_T(&y, 2);

  /*  EM iterations  */
  /* 'g_mix_est_fast_lik:10' min_sg=0.1; */
  /* 'g_mix_est_fast_lik:12' raw_sigvoc=max(sigv.^2, min_sg^2); */
  i14 = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = sigv->size[1];
  emxEnsureCapacity_real_T(y, i14);
  nx = sigv->size[1];
  for (k = 0; k < nx; k++) {
    y->data[k] = rt_powd_snf(sigv->data[k], 2.0);
  }

  emxInit_real_T(&raw_sigvoc, 2);
  i14 = raw_sigvoc->size[0] * raw_sigvoc->size[1];
  raw_sigvoc->size[0] = 1;
  raw_sigvoc->size[1] = y->size[1];
  emxEnsureCapacity_real_T(raw_sigvoc, i14);
  nx = y->size[1];
  for (k = 0; k < nx; k++) {
    change = y->data[k];
    if (!(change > 0.010000000000000002)) {
      change = 0.010000000000000002;
    }

    raw_sigvoc->data[k] = change;
  }

  /*  ppoc=pp; */
  /* 'g_mix_est_fast_lik:15' sample = raw_sample(:)'; */
  /*  make vectors verticsl */
  /*  if size(raw_sample,1) > 1 */
  /*      sample=raw_sample'; */
  /*  else */
  /*      sample = raw_sample; */
  /*  end */
  /* 'g_mix_est_fast_lik:23' mivoc = muv(:); */
  i14 = musort->size[0];
  musort->size[0] = muv->size[1];
  emxEnsureCapacity_real_T(musort, i14);
  loop_ub = muv->size[1];
  for (i14 = 0; i14 < loop_ub; i14++) {
    musort->data[i14] = muv->data[i14];
  }

  emxInit_real_T(&sigvoc, 1);

  /*  if size(muv,2) > 1 */
  /*      mivoc=muv'; */
  /*  else */
  /*      mivoc = muv; */
  /*  end */
  /* 'g_mix_est_fast_lik:30' sigvoc = raw_sigvoc(:); */
  i14 = sigvoc->size[0];
  sigvoc->size[0] = raw_sigvoc->size[1];
  emxEnsureCapacity_real_T(sigvoc, i14);
  loop_ub = raw_sigvoc->size[1];
  for (i14 = 0; i14 < loop_ub; i14++) {
    sigvoc->data[i14] = raw_sigvoc->data[i14];
  }

  emxFree_real_T(&raw_sigvoc);
  emxInit_real_T(&ppoc, 1);

  /*  if size(raw_sigvoc,2) > 1 */
  /*      sigvoc=raw_sigvoc'; */
  /*  else */
  /*      sigvoc = raw_sigvoc; */
  /*  end */
  /* 'g_mix_est_fast_lik:37' ppoc = pp(:); */
  i14 = ppoc->size[0];
  ppoc->size[0] = pp->size[1];
  emxEnsureCapacity_real_T(ppoc, i14);
  loop_ub = pp->size[1];
  for (i14 = 0; i14 < loop_ub; i14++) {
    ppoc->data[i14] = pp->data[i14];
  }

  emxInit_real_T(&pssmac, 2);

  /*  if size(pp,2) > 1 */
  /*      ppoc=pp'; */
  /*  else */
  /*      ppoc = pp; */
  /*  end */
  /* 'g_mix_est_fast_lik:46' N=length(sample); */
  unnamed_idx_1 = (unsigned int)raw_sample->size[0];
  varargin_1 = (int)unnamed_idx_1;

  /*  OK oceniamy iteracyjnie wg wzorow z artykulu Bilmsa */
  /* 'g_mix_est_fast_lik:49' pssmac=zeros(KS,N); */
  unnamed_idx_1 = (unsigned int)raw_sample->size[0];
  b_unnamed_idx_1 = (int)unnamed_idx_1;
  i14 = pssmac->size[0] * pssmac->size[1];
  loop_ub = (int)KS;
  pssmac->size[0] = loop_ub;
  pssmac->size[1] = (int)unnamed_idx_1;
  emxEnsureCapacity_real_T(pssmac, i14);
  vlen = loop_ub * (int)unnamed_idx_1;
  for (i14 = 0; i14 < vlen; i14++) {
    pssmac->data[i14] = 0.0;
  }

  /* 'g_mix_est_fast_lik:50' change=1; */
  change = 1.0;

  /* 'g_mix_est_fast_lik:51' rescue_iterations_count = 0; */
  rescue_iterations_count = 0;

  /* 'g_mix_est_fast_lik:52' unreachable_number_of_iterations = 10000; */
  /* 'g_mix_est_fast_lik:54' while change > 1.5e-4 && rescue_iterations_count < unreachable_number_of_iterations */
  emxInit_real_T(&oldppoc, 1);
  emxInit_real_T(&oldsigvoc, 1);
  emxInit_real_T(&pskmac, 2);
  emxInit_real_T(&ppp, 1);
  emxInit_real_T(&sigmac, 2);
  emxInit_real_T(&b, 1);
  emxInit_real_T(&r0, 2);
  emxInit_real_T(&b_musort, 2);
  emxInit_real_T(&r1, 2);
  emxInit_real_T(&c_musort, 2);
  emxInit_real_T(&d_musort, 1);
  while ((change > 0.00015) && (rescue_iterations_count < 10000)) {
    /* 'g_mix_est_fast_lik:55' rescue_iterations_count = rescue_iterations_count + 1; */
    rescue_iterations_count++;

    /* 'g_mix_est_fast_lik:56' oldppoc=ppoc; */
    i14 = oldppoc->size[0];
    oldppoc->size[0] = ppoc->size[0];
    emxEnsureCapacity_real_T(oldppoc, i14);
    vlen = ppoc->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      oldppoc->data[i14] = ppoc->data[i14];
    }

    /* 'g_mix_est_fast_lik:57' oldsigvoc=sigvoc; */
    i14 = oldsigvoc->size[0];
    oldsigvoc->size[0] = sigvoc->size[0];
    emxEnsureCapacity_real_T(oldsigvoc, i14);
    vlen = sigvoc->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      oldsigvoc->data[i14] = sigvoc->data[i14];
    }

    /*  lower limit for component weights */
    /* 'g_mix_est_fast_lik:60' ppoc=max(ppoc,0.001); */
    nx = ppoc->size[0];
    unnamed_idx_1 = (unsigned int)ppoc->size[0];
    i14 = ppoc->size[0];
    ppoc->size[0] = nx;
    emxEnsureCapacity_real_T(ppoc, i14);
    nx = (int)unnamed_idx_1;
    for (k = 0; k < nx; k++) {
      change = oldppoc->data[k];
      if (!(change > 0.001)) {
        change = 0.001;
      }

      ppoc->data[k] = change;
    }

    /* 'g_mix_est_fast_lik:62' for kskla=1:KS */
    i14 = pssmac->size[0] * pssmac->size[1];
    pssmac->size[0] = loop_ub;
    pssmac->size[1] = b_unnamed_idx_1;
    emxEnsureCapacity_real_T(pssmac, i14);
    for (nx = 0; nx < loop_ub; nx++) {
      /* 'g_mix_est_fast_lik:63' pssmac(kskla,:)=ppoc(kskla)*normpdf(sample,mivoc(kskla),sqrt(sigvoc(kskla))); */
      change = sqrt(sigvoc->data[nx]);
      unnamed_idx_1 = (unsigned int)raw_sample->size[0];
      i14 = y->size[0] * y->size[1];
      y->size[0] = 1;
      y->size[1] = (int)unnamed_idx_1;
      emxEnsureCapacity_real_T(y, i14);
      i14 = y->size[1];
      for (k = 0; k < i14; k++) {
        if (change > 0.0) {
          t = (raw_sample->data[k] - musort->data[nx]) / change;
          y->data[k] = exp(-0.5 * t * t) / (2.5066282746310002 * change);
        } else {
          y->data[k] = rtNaN;
        }
      }

      change = ppoc->data[nx];
      vlen = y->size[1];
      for (i14 = 0; i14 < vlen; i14++) {
        pssmac->data[nx + pssmac->size[0] * i14] = change * y->data[i14];
      }
    }

    /* 'g_mix_est_fast_lik:65' psummac=ones(KS,1)*sum(pssmac,1); */
    /* 'g_mix_est_fast_lik:66' pskmac=pssmac./psummac; */
    c_sum(pssmac, y);
    i14 = pskmac->size[0] * pskmac->size[1];
    pskmac->size[0] = loop_ub;
    pskmac->size[1] = y->size[1];
    emxEnsureCapacity_real_T(pskmac, i14);
    for (i14 = 0; i14 < loop_ub; i14++) {
      vlen = y->size[1];
      for (nx = 0; nx < vlen; nx++) {
        change = y->data[nx];
        pskmac->data[i14 + pskmac->size[0] * nx] = pssmac->data[i14 +
          pssmac->size[0] * nx] / change;
      }
    }

    /* 'g_mix_est_fast_lik:67' ppp=sum(pskmac,2); */
    vlen = pskmac->size[1];
    if (pskmac->size[1] == 0) {
      nx = pskmac->size[0];
      i14 = ppp->size[0];
      ppp->size[0] = nx;
      emxEnsureCapacity_real_T(ppp, i14);
      for (i14 = 0; i14 < nx; i14++) {
        ppp->data[i14] = 0.0;
      }
    } else {
      vstride = pskmac->size[0];
      i14 = ppp->size[0];
      ppp->size[0] = pskmac->size[0];
      emxEnsureCapacity_real_T(ppp, i14);
      for (nx = 0; nx < vstride; nx++) {
        ppp->data[nx] = pskmac->data[nx];
      }

      for (k = 2; k <= vlen; k++) {
        xoffset = (k - 1) * vstride;
        for (nx = 0; nx < vstride; nx++) {
          ppp->data[nx] += pskmac->data[xoffset + nx];
        }
      }
    }

    /* 'g_mix_est_fast_lik:68' ppoc=ppp/N; */
    i14 = ppoc->size[0];
    ppoc->size[0] = ppp->size[0];
    emxEnsureCapacity_real_T(ppoc, i14);
    vlen = ppp->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      ppoc->data[i14] = ppp->data[i14] / (double)varargin_1;
    }

    /* 'g_mix_est_fast_lik:69' mivoc=pskmac*sample'; */
    if ((pskmac->size[1] == 1) || (raw_sample->size[0] == 1)) {
      i14 = musort->size[0];
      musort->size[0] = pskmac->size[0];
      emxEnsureCapacity_real_T(musort, i14);
      vlen = pskmac->size[0];
      for (i14 = 0; i14 < vlen; i14++) {
        musort->data[i14] = 0.0;
        xoffset = pskmac->size[1];
        for (nx = 0; nx < xoffset; nx++) {
          musort->data[i14] += pskmac->data[i14 + pskmac->size[0] * nx] *
            raw_sample->data[nx];
        }
      }
    } else {
      vlen = pskmac->size[0];
      vstride = pskmac->size[1];
      i14 = musort->size[0];
      musort->size[0] = pskmac->size[0];
      emxEnsureCapacity_real_T(musort, i14);
      for (nx = 0; nx < vlen; nx++) {
        musort->data[nx] = 0.0;
      }

      for (k = 0; k < vstride; k++) {
        xoffset = k * vlen;
        for (nx = 0; nx < vlen; nx++) {
          musort->data[nx] += raw_sample->data[k] * pskmac->data[xoffset + nx];
        }
      }
    }

    /* 'g_mix_est_fast_lik:70' mivoc=mivoc./ppp; */
    i14 = d_musort->size[0];
    d_musort->size[0] = musort->size[0];
    emxEnsureCapacity_real_T(d_musort, i14);
    vlen = musort->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      d_musort->data[i14] = musort->data[i14];
    }

    rdivide_helper(d_musort, ppp, musort);

    /* 'g_mix_est_fast_lik:71' sigmac=(ones(KS,1)*sample-mivoc*ones(1,N)).*((ones(KS,1)*sample-mivoc*ones(1,N))); */
    i14 = r0->size[0] * r0->size[1];
    r0->size[0] = loop_ub;
    r0->size[1] = raw_sample->size[0];
    emxEnsureCapacity_real_T(r0, i14);
    for (i14 = 0; i14 < loop_ub; i14++) {
      vlen = raw_sample->size[0];
      for (nx = 0; nx < vlen; nx++) {
        r0->data[i14 + r0->size[0] * nx] = raw_sample->data[nx];
      }
    }

    i14 = b_musort->size[0] * b_musort->size[1];
    b_musort->size[0] = musort->size[0];
    b_musort->size[1] = varargin_1;
    emxEnsureCapacity_real_T(b_musort, i14);
    vlen = musort->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      for (nx = 0; nx < varargin_1; nx++) {
        b_musort->data[i14 + b_musort->size[0] * nx] = musort->data[i14];
      }
    }

    i14 = r1->size[0] * r1->size[1];
    r1->size[0] = loop_ub;
    r1->size[1] = raw_sample->size[0];
    emxEnsureCapacity_real_T(r1, i14);
    for (i14 = 0; i14 < loop_ub; i14++) {
      vlen = raw_sample->size[0];
      for (nx = 0; nx < vlen; nx++) {
        r1->data[i14 + r1->size[0] * nx] = raw_sample->data[nx];
      }
    }

    i14 = c_musort->size[0] * c_musort->size[1];
    c_musort->size[0] = musort->size[0];
    c_musort->size[1] = varargin_1;
    emxEnsureCapacity_real_T(c_musort, i14);
    vlen = musort->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      for (nx = 0; nx < varargin_1; nx++) {
        c_musort->data[i14 + c_musort->size[0] * nx] = musort->data[i14];
      }
    }

    i14 = sigmac->size[0] * sigmac->size[1];
    sigmac->size[0] = r0->size[0];
    sigmac->size[1] = r0->size[1];
    emxEnsureCapacity_real_T(sigmac, i14);
    vlen = r0->size[0] * r0->size[1];
    for (i14 = 0; i14 < vlen; i14++) {
      sigmac->data[i14] = (r0->data[i14] - b_musort->data[i14]) * (r1->data[i14]
        - c_musort->data[i14]);
    }

    /* 'g_mix_est_fast_lik:72' for kkk=1:KS */
    for (nx = 0; nx < loop_ub; nx++) {
      /*  lower limit for component variances  */
      /* 'g_mix_est_fast_lik:74' sigvoc(kkk)=max([pskmac(kkk,:)*sigmac(kkk,:)' min_sg^2]); */
      vlen = sigmac->size[1];
      i14 = b->size[0];
      b->size[0] = vlen;
      emxEnsureCapacity_real_T(b, i14);
      for (i14 = 0; i14 < vlen; i14++) {
        b->data[i14] = sigmac->data[nx + sigmac->size[0] * i14];
      }

      i14 = pskmac->size[1];
      if ((i14 == 1) || (b->size[0] == 1)) {
        vlen = pskmac->size[1];
        change = 0.0;
        for (i14 = 0; i14 < vlen; i14++) {
          change += pskmac->data[nx + pskmac->size[0] * i14] * b->data[i14];
        }
      } else {
        vlen = pskmac->size[1];
        change = 0.0;
        for (i14 = 0; i14 < vlen; i14++) {
          change += pskmac->data[nx + pskmac->size[0] * i14] * b->data[i14];
        }
      }

      if ((change < 0.010000000000000002) || rtIsNaN(change)) {
        sigvoc->data[nx] = 0.010000000000000002;
      } else {
        sigvoc->data[nx] = change;
      }
    }

    /* 'g_mix_est_fast_lik:76' sigvoc=sigvoc./ppp; */
    i14 = d_musort->size[0];
    d_musort->size[0] = sigvoc->size[0];
    emxEnsureCapacity_real_T(d_musort, i14);
    vlen = sigvoc->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      d_musort->data[i14] = sigvoc->data[i14];
    }

    rdivide_helper(d_musort, ppp, sigvoc);

    /*  */
    /* 'g_mix_est_fast_lik:79' change=sum(abs(ppoc-oldppoc))+sum(abs(sigvoc-oldsigvoc))/KS; */
    i14 = oldppoc->size[0];
    oldppoc->size[0] = ppoc->size[0];
    emxEnsureCapacity_real_T(oldppoc, i14);
    vlen = ppoc->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      oldppoc->data[i14] = ppoc->data[i14] - oldppoc->data[i14];
    }

    b_abs(oldppoc, ppp);
    i14 = oldsigvoc->size[0];
    oldsigvoc->size[0] = sigvoc->size[0];
    emxEnsureCapacity_real_T(oldsigvoc, i14);
    vlen = sigvoc->size[0];
    for (i14 = 0; i14 < vlen; i14++) {
      oldsigvoc->data[i14] = sigvoc->data[i14] - oldsigvoc->data[i14];
    }

    b_abs(oldsigvoc, oldppoc);
    change = sum(ppp) + sum(oldppoc) / KS;
  }

  emxFree_real_T(&d_musort);
  emxFree_real_T(&c_musort);
  emxFree_real_T(&r1);
  emxFree_real_T(&b_musort);
  emxFree_real_T(&r0);
  emxFree_real_T(&b);
  emxFree_real_T(&sigmac);
  emxFree_real_T(&ppp);
  emxFree_real_T(&pskmac);
  emxFree_real_T(&oldsigvoc);

  /*  compute likelihood */
  /* 'g_mix_est_fast_lik:83' l_lik=sum(log(sum(pssmac,1))); */
  c_sum(pssmac, y);
  nx = y->size[1];
  emxFree_real_T(&pssmac);
  for (k = 0; k < nx; k++) {
    y->data[k] = log(y->data[k]);
  }

  emxInit_int32_T(&iidx, 1);
  *l_lik = b_sum(y);

  /*  sort estimates */
  /* 'g_mix_est_fast_lik:86' [musort,isort]=sort(mivoc); */
  sort(musort, iidx);
  i14 = oldppoc->size[0];
  oldppoc->size[0] = iidx->size[0];
  emxEnsureCapacity_real_T(oldppoc, i14);
  loop_ub = iidx->size[0];
  emxFree_real_T(&y);
  for (i14 = 0; i14 < loop_ub; i14++) {
    oldppoc->data[i14] = iidx->data[i14];
  }

  emxFree_int32_T(&iidx);

  /* 'g_mix_est_fast_lik:87' sigsort=sqrt(sigvoc(isort)); */
  i14 = sigsort->size[0];
  sigsort->size[0] = oldppoc->size[0];
  emxEnsureCapacity_real_T(sigsort, i14);
  loop_ub = oldppoc->size[0];
  for (i14 = 0; i14 < loop_ub; i14++) {
    sigsort->data[i14] = sigvoc->data[(int)oldppoc->data[i14] - 1];
  }

  emxFree_real_T(&sigvoc);
  nx = oldppoc->size[0];
  for (k = 0; k < nx; k++) {
    sigsort->data[k] = sqrt(sigsort->data[k]);
  }

  /* 'g_mix_est_fast_lik:88' ppsort=ppoc(isort); */
  i14 = ppsort->size[0];
  ppsort->size[0] = oldppoc->size[0];
  emxEnsureCapacity_real_T(ppsort, i14);
  loop_ub = oldppoc->size[0];
  for (i14 = 0; i14 < loop_ub; i14++) {
    ppsort->data[i14] = ppoc->data[(int)oldppoc->data[i14] - 1];
  }

  emxFree_real_T(&oldppoc);
  emxFree_real_T(&ppoc);
}

/*
 * File trailer for g_mix_est_fast_lik.c
 *
 * [EOF]
 */
