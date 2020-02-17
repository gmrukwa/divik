/*
 *
 * fetch_thresholds_types.h
 *
 * Code generation for function 'fetch_thresholds_types'
 *
 */

#ifndef FETCH_THRESHOLDS_TYPES_H
#define FETCH_THRESHOLDS_TYPES_H

/* Include files */
#include "rtwtypes.h"

/* Type Definitions */
#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T

struct emxArray_real_T
{
  double *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  bool canFreeData;
};

#endif                                 /*struct_emxArray_real_T*/

#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T

typedef struct emxArray_real_T emxArray_real_T;

#endif                                 /*typedef_emxArray_real_T*/

#ifndef typedef_cell_wrap_0
#define typedef_cell_wrap_0

typedef struct {
  emxArray_real_T *f1;
} cell_wrap_0;

#endif                                 /*typedef_cell_wrap_0*/

#ifndef typedef_cell_wrap_2
#define typedef_cell_wrap_2

typedef struct {
  double f1[2];
} cell_wrap_2;

#endif                                 /*typedef_cell_wrap_2*/

#ifndef struct_emxArray_boolean_T
#define struct_emxArray_boolean_T

struct emxArray_boolean_T
{
  bool *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  bool canFreeData;
};

#endif                                 /*struct_emxArray_boolean_T*/

#ifndef typedef_emxArray_boolean_T
#define typedef_emxArray_boolean_T

typedef struct emxArray_boolean_T emxArray_boolean_T;

#endif                                 /*typedef_emxArray_boolean_T*/

#ifndef typedef_emxArray_cell_wrap_0
#define typedef_emxArray_cell_wrap_0

typedef struct {
  cell_wrap_0 *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  bool canFreeData;
} emxArray_cell_wrap_0;

#endif                                 /*typedef_emxArray_cell_wrap_0*/

#ifndef struct_emxArray_int32_T
#define struct_emxArray_int32_T

struct emxArray_int32_T
{
  int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  bool canFreeData;
};

#endif                                 /*struct_emxArray_int32_T*/

#ifndef typedef_emxArray_int32_T
#define typedef_emxArray_int32_T

typedef struct emxArray_int32_T emxArray_int32_T;

#endif                                 /*typedef_emxArray_int32_T*/

#ifndef struct_emxArray_uint32_T
#define struct_emxArray_uint32_T

struct emxArray_uint32_T
{
  unsigned int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  bool canFreeData;
};

#endif                                 /*struct_emxArray_uint32_T*/

#ifndef typedef_emxArray_uint32_T
#define typedef_emxArray_uint32_T

typedef struct emxArray_uint32_T emxArray_uint32_T;

#endif                                 /*typedef_emxArray_uint32_T*/
#endif

/* End of code generation (fetch_thresholds_types.h) */
