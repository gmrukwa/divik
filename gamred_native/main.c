/*
 * File: main.c
 *
 * MATLAB Coder version            : 4.2
 * C/C++ source code generated on  : 23-Dec-2019 23:22:36
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/
/* Include Files */
#include <Python.h>
#include "numpy/arrayobject.h"
#include <string.h>
#include "rt_nonfinite.h"
#include "fetch_thresholds.h"
#include "main.h"
#include "fetch_thresholds_terminate.h"
#include "fetch_thresholds_emxAPI.h"
#include "fetch_thresholds_initialize.h"

static emxArray_real_T *PyArrayObject_to_emxArray(PyArrayObject *vals)
{
  int iv0[1] = { PyArray_SIZE(vals) };
  emxArray_real_T *result = emxCreateND_real_T(1, iv0);
  memcpy(result->data, PyArray_DATA(vals), PyArray_NBYTES(vals));
  return result;
}

static PyArrayObject *emxArray_to_PyArrayObject(emxArray_real_T *vals)
{
  npy_intp iv0[] = { vals->size[0U] };
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, iv0, NPY_DOUBLE);
  memcpy(PyArray_DATA(result), vals->data, PyArray_NBYTES(result));
  return result;
}

static PyObject *method_find_thresholds(PyObject *self, PyObject *args) {
    PyObject *vals=NULL;
    PyArrayObject *vals_arr=NULL;
    PyArrayObject *thresholds=NULL;
    unsigned long max_components = 0;
    emxArray_real_T *native_thresholds;
    emxArray_real_T *native_vals;

    /* Parse arguments */
    // https://docs.python.org/3/c-api/arg.html
    if(!PyArg_ParseTuple(args, "Ok", &vals, &max_components)) {
        return NULL;
    }

    vals_arr = (PyArrayObject *)PyArray_FROM_OTF(vals, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (vals_arr == NULL) {
        return NULL;
    }

    /* Initialize the application.
      You do not need to do this more than one time. */
    fetch_thresholds_initialize();

    emxInitArray_real_T(&native_thresholds, 1);
    native_vals = PyArrayObject_to_emxArray(vals_arr);
    Py_DECREF(vals_arr);

    fetch_thresholds(native_vals, max_components, native_thresholds);

    emxDestroyArray_real_T(native_vals);
    thresholds = emxArray_to_PyArrayObject(native_thresholds);
    emxDestroyArray_real_T(native_thresholds);
    
    /* Terminate the application.
      You do not need to do this more than one time. */
    fetch_thresholds_terminate();

    return (PyObject*)thresholds;
}

static PyMethodDef GamredNativeMethods[] = {
    {"find_thresholds", method_find_thresholds, METH_VARARGS, "Python interface for the MATLAB fetch_thresholds function"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef gamred_native_module = {
    PyModuleDef_HEAD_INIT,
    "gamred_native",
    "Python interface for the MATLAB fetch_thresholds function",
    -1,
    GamredNativeMethods
};

PyMODINIT_FUNC PyInit_gamred_native(void) {
    import_array();
    return PyModule_Create(&gamred_native_module);
}
