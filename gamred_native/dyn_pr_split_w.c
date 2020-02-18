/*
 *
 * dyn_pr_split_w.c
 *
 * Code generation for function 'dyn_pr_split_w'
 *
 */

/* Include files */
#include "dyn_pr_split_w.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "my_qu_ix_w.h"
#include "rt_nonfinite.h"

/* Function Definitions */

/*
 * function [Q,opt_part]=dyn_pr_split_w(data,ygreki,K_gr,aux_mx)
 */
void dyn_pr_split_w(const emxArray_real_T *data, const emxArray_real_T *ygreki,
                    double K_gr, const emxArray_real_T *aux_mx, emxArray_real_T *
                    Q, emxArray_real_T *opt_part)
{
  emxArray_real_T *p_opt_idx;
  emxArray_real_T *p_aux;
  int i;
  int loop_ub;
  int varargin_1;
  int N;
  int b_loop_ub;
  emxArray_uint32_T *opt_pals;
  emxArray_real_T *b_data;
  emxArray_real_T *b_ygreki;
  int kk;
  int i1;
  int i2;
  int p_aux_tmp;
  emxArray_real_T *x;
  int n_tmp;
  int kster;
  double ex;
  double opt_part_tmp;
  bool exitg1;
  emxInit_real_T(&p_opt_idx, 2);
  emxInit_real_T(&p_aux, 2);

  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  */
  /*         main dynamic programming algorithm */
  /*  */
  /*  initialize */
  /* 'dyn_pr_split_w:9' Q=zeros(1,K_gr); */
  i = Q->size[0] * Q->size[1];
  Q->size[0] = 1;
  loop_ub = (int)K_gr;
  Q->size[1] = loop_ub;
  emxEnsureCapacity_real_T(Q, i);

  /* 'dyn_pr_split_w:10' N=length(data); */
  varargin_1 = data->size[1] - 1;
  N = data->size[1];

  /* 'dyn_pr_split_w:11' p_opt_idx=zeros(1,N); */
  i = p_opt_idx->size[0] * p_opt_idx->size[1];
  p_opt_idx->size[0] = 1;
  p_opt_idx->size[1] = data->size[1];
  emxEnsureCapacity_real_T(p_opt_idx, i);

  /* 'dyn_pr_split_w:12' p_aux=zeros(1,N); */
  i = p_aux->size[0] * p_aux->size[1];
  p_aux->size[0] = 1;
  p_aux->size[1] = data->size[1];
  emxEnsureCapacity_real_T(p_aux, i);
  b_loop_ub = data->size[1];
  for (i = 0; i < b_loop_ub; i++) {
    p_aux->data[i] = 0.0;
  }

  emxInit_uint32_T(&opt_pals, 2);

  /* 'dyn_pr_split_w:13' opt_pals=ones(K_gr,N); */
  i = opt_pals->size[0] * opt_pals->size[1];
  opt_pals->size[0] = loop_ub;
  opt_pals->size[1] = data->size[1];
  emxEnsureCapacity_uint32_T(opt_pals, i);
  b_loop_ub = loop_ub * data->size[1];
  for (i = 0; i < b_loop_ub; i++) {
    opt_pals->data[i] = 1U;
  }

  /* 'dyn_pr_split_w:14' for kk=1:N */
  i = data->size[1];
  emxInit_real_T(&b_data, 2);
  emxInit_real_T(&b_ygreki, 2);
  for (kk = 0; kk < i; kk++) {
    /* 'dyn_pr_split_w:15' p_opt_idx(kk)=my_qu_ix_w(data(kk:N),ygreki(kk:N)); */
    if (kk + 1 > N) {
      i1 = 0;
      i2 = -1;
      p_aux_tmp = 0;
      n_tmp = -1;
    } else {
      i1 = kk;
      i2 = varargin_1;
      p_aux_tmp = kk;
      n_tmp = varargin_1;
    }

    kster = b_data->size[0] * b_data->size[1];
    b_data->size[0] = 1;
    b_loop_ub = i2 - i1;
    b_data->size[1] = b_loop_ub + 1;
    emxEnsureCapacity_real_T(b_data, kster);
    for (i2 = 0; i2 <= b_loop_ub; i2++) {
      b_data->data[i2] = data->data[i1 + i2];
    }

    i1 = b_ygreki->size[0] * b_ygreki->size[1];
    b_ygreki->size[0] = 1;
    b_loop_ub = n_tmp - p_aux_tmp;
    b_ygreki->size[1] = b_loop_ub + 1;
    emxEnsureCapacity_real_T(b_ygreki, i1);
    for (i1 = 0; i1 <= b_loop_ub; i1++) {
      b_ygreki->data[i1] = ygreki->data[p_aux_tmp + i1];
    }

    p_opt_idx->data[kk] = my_qu_ix_w(b_data, b_ygreki);
  }

  emxFree_real_T(&b_ygreki);
  emxFree_real_T(&b_data);

  /*  aux_mx - already computed */
  /*  iterate */
  /* 'dyn_pr_split_w:21' for kster=1:K_gr */
  emxInit_real_T(&x, 2);
  for (kster = 0; kster < loop_ub; kster++) {
    /* 'dyn_pr_split_w:22' for kk=1:N-kster */
    i = N - kster;
    for (kk = 0; kk <= i - 2; kk++) {
      /* 'dyn_pr_split_w:23' for jj=kk+1:N-kster+1 */
      i1 = i - kk;
      for (b_loop_ub = 0; b_loop_ub <= i1 - 2; b_loop_ub++) {
        /* 'dyn_pr_split_w:24' p_aux(jj)= aux_mx(kk,jj)+p_opt_idx(jj); */
        p_aux_tmp = (int)((((double)kk + 1.0) + 1.0) + (double)b_loop_ub) - 1;
        p_aux->data[p_aux_tmp] = aux_mx->data[kk + aux_mx->size[0] * p_aux_tmp]
          + p_opt_idx->data[p_aux_tmp];
      }

      /* 'dyn_pr_split_w:26' [mm,ix]=min(p_aux(kk+1:N-kster+1)); */
      i1 = (int)((double)(varargin_1 + 1) - ((double)kster + 1.0));
      if (((double)kk + 1.0) + 1.0 > i1 + 1) {
        i2 = -1;
        i1 = -1;
      } else {
        i2 = kk;
      }

      b_loop_ub = i1 - i2;
      n_tmp = (unsigned short)b_loop_ub;
      if (n_tmp <= 2) {
        if (n_tmp == 1) {
          p_opt_idx->data[kk] = p_aux->data[i2 + 1];
          p_aux_tmp = 1;
        } else {
          opt_part_tmp = p_aux->data[i2 + 1];
          ex = p_aux->data[i2 + 2];
          if ((opt_part_tmp > ex) || (rtIsNaN(opt_part_tmp) && (!rtIsNaN(ex))))
          {
            p_opt_idx->data[kk] = ex;
            p_aux_tmp = 2;
          } else {
            p_opt_idx->data[kk] = opt_part_tmp;
            p_aux_tmp = 1;
          }
        }
      } else {
        i1 = x->size[0] * x->size[1];
        x->size[0] = 1;
        x->size[1] = b_loop_ub;
        emxEnsureCapacity_real_T(x, i1);
        for (i1 = 0; i1 < b_loop_ub; i1++) {
          x->data[i1] = p_aux->data[(i2 + i1) + 1];
        }

        opt_part_tmp = p_aux->data[i2 + 1];
        if (!rtIsNaN(opt_part_tmp)) {
          p_aux_tmp = 1;
        } else {
          p_aux_tmp = 0;
          b_loop_ub = 2;
          exitg1 = false;
          while ((!exitg1) && (b_loop_ub <= n_tmp)) {
            if (!rtIsNaN(x->data[b_loop_ub - 1])) {
              p_aux_tmp = b_loop_ub;
              exitg1 = true;
            } else {
              b_loop_ub++;
            }
          }
        }

        if (p_aux_tmp == 0) {
          p_opt_idx->data[kk] = opt_part_tmp;
          p_aux_tmp = 1;
        } else {
          ex = p_aux->data[i2 + p_aux_tmp];
          i1 = p_aux_tmp + 1;
          for (b_loop_ub = i1; b_loop_ub <= n_tmp; b_loop_ub++) {
            opt_part_tmp = p_aux->data[i2 + b_loop_ub];
            if (ex > opt_part_tmp) {
              ex = opt_part_tmp;
              p_aux_tmp = b_loop_ub;
            }
          }

          p_opt_idx->data[kk] = ex;
        }
      }

      /* 'dyn_pr_split_w:27' p_opt_idx(kk)=mm; */
      /* 'dyn_pr_split_w:28' opt_pals(kster,kk)=kk+ix(1); */
      opt_pals->data[kster + opt_pals->size[0] * kk] = ((unsigned int)kk +
        p_aux_tmp) + 1U;
    }

    /* 'dyn_pr_split_w:30' Q(kster)=p_opt_idx(1); */
    Q->data[kster] = p_opt_idx->data[0];
  }

  emxFree_real_T(&x);
  emxFree_real_T(&p_aux);
  emxFree_real_T(&p_opt_idx);

  /*  restore optimal decisions */
  /* 'dyn_pr_split_w:35' opt_part=zeros(1,K_gr); */
  i = opt_part->size[0] * opt_part->size[1];
  opt_part->size[0] = 1;
  opt_part->size[1] = loop_ub;
  emxEnsureCapacity_real_T(opt_part, i);
  for (i = 0; i < loop_ub; i++) {
    opt_part->data[i] = 0.0;
  }

  /* 'dyn_pr_split_w:36' opt_part(1)=opt_pals(K_gr,1); */
  opt_part->data[0] = opt_pals->data[loop_ub - 1];

  /* 'dyn_pr_split_w:37' for kster=K_gr-1:-1:1 */
  i = (int)(((-1.0 - (K_gr - 1.0)) + 1.0) / -1.0);
  for (kster = 0; kster < i; kster++) {
    ex = (K_gr - 1.0) + -(double)kster;

    /*  	if K_gr-kster+1<=0 */
    /*  		fprintf('K_gr-kster+1<=0\n'); */
    /*  	elseif opt_part(K_gr-kster)<=0 */
    /*  		fprintf('opt_part(K_gr-kster)<=0\n'); */
    /*  	end */
    /* 'dyn_pr_split_w:43' opt_part(K_gr-kster+1)=opt_pals(kster,opt_part(K_gr-kster)); */
    opt_part_tmp = K_gr - ex;
    opt_part->data[(int)(opt_part_tmp + 1.0) - 1] = opt_pals->data[((int)ex +
      opt_pals->size[0] * ((int)(unsigned int)opt_part->data[(int)opt_part_tmp -
      1] - 1)) - 1];
  }

  emxFree_uint32_T(&opt_pals);
}

/* End of code generation (dyn_pr_split_w.c) */
