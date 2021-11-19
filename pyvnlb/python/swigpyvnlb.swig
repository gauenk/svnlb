
// Chunks copied (basically) verbatim from FAISS


// -*- C++ -*-

// This file describes the C++-scripting language bridge for Python (and formerly Lua).
// It contains mainly includes and a few macros. There are
// 2 preprocessor macros of interest:

// SWIGPYTHON: Python-specific code
// GPU_WRAPPER: also compile interfaces for GPU.

%module swigpyvnlb;

// NOTE: While parsing the headers to generate the interface, SWIG does not know
// about `_MSC_VER`.
// TODO: Remove the need for this hack.
#ifdef SWIGWIN
#define _MSC_VER
%include <windows.i>
#endif // SWIGWIN

// fbode SWIG fails on warnings, so make them non fatal
#pragma SWIG nowarn=321
#pragma SWIG nowarn=403
#pragma SWIG nowarn=325
#pragma SWIG nowarn=389
#pragma SWIG nowarn=341
#pragma SWIG nowarn=512
#pragma SWIG nowarn=362

%include <stdint.i>

typedef uint64_t size_t;


#define __restrict

/*******************************************************************
 * Copied verbatim to wrapper. Contains the C++-visible includes, and
 * the language includes for their respective matrix libraries.
 *******************************************************************/

%{


#include <stdint.h>
#include <omp.h>

#ifdef SWIGPYTHON

#undef popcount64

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#endif

#include <pyvnlb/cpp/utils/VnlbVersion.h>
#include <pyvnlb/cpp/utils/VnlbInterrupt.h>
#include <pyvnlb/cpp/utils/VnlbAsserts.h>
#include <pyvnlb/cpp/video_io/io.h>

%}


/********************************************************
 * GIL manipulation and exception handling
 ********************************************************/

#ifdef SWIGPYTHON
// %catches(vnlb::VnlbException);


// Python-specific: release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(vnlb::VnlbException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc & ba) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    } catch(const std::exception& ex) {
        PyEval_RestoreThread(_save);
        std::string what = std::string("C++ exception ") + ex.what();
        PyErr_SetString(PyExc_RuntimeError, what.c_str());
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

#endif

/*******************************************************************
 * Parse headers
 *******************************************************************/

%include <pyvnlb/cpp/utils/platform_macros.h>

%ignore *::cmp;

%{

#include <pyvnlb/cpp/pybind/video_io/interface.h>
#include <pyvnlb/cpp/pybind/flow/interface.h>
#include <pyvnlb/cpp/pybind/vnlb/interface.h>

%}

%include <pyvnlb/cpp/pybind/video_io/interface.h>
%include <pyvnlb/cpp/pybind/flow/interface.h>
%include <pyvnlb/cpp/pybind/vnlb/interface.h>
%include <pyvnlb/cpp/video_io/io.h>
%include <pyvnlb/cpp/utils/VnlbVersion.h>


%ignore vnlb::InterruptCallback::instance;
%ignore vnlb::InterruptCallback::lock;


/*******************************************************************
 * Support I/O to arbitrary functions
 *******************************************************************/

#ifdef SWIGPYTHON
%include <pyvnlb/python/python_callbacks.h>


%{
#include <pyvnlb/python/python_callbacks.h>
%}

#endif

// Python-specific: do not release GIL any more, as functions below
// use the Python/C API
#ifdef SWIGPYTHON
%exception;
#endif


/*******************************************************************
 * Python specific: numpy array <-> C++ pointer interface
 *******************************************************************/

#ifdef SWIGPYTHON

%{
PyObject *swig_ptr (PyObject *a)
{

    if (PyBytes_Check(a)) {
        return SWIG_NewPointerObj(PyBytes_AsString(a), SWIGTYPE_p_char, 0);
    }
    if (PyByteArray_Check(a)) {
        return SWIG_NewPointerObj(PyByteArray_AsString(a), SWIGTYPE_p_char, 0);
    }
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_BOOL) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_bool, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long_long, 0);
#endif
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long_long, 0);
#endif
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}


struct PythonInterruptCallback: vnlb::InterruptCallback {

    bool want_interrupt () override {
        int err;
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            err = PyErr_CheckSignals();
            PyGILState_Release(gstate);
        }
        return err == -1;
    }

};


%}


%init %{
    /* needed, else crash at runtime */
    import_array();

    vnlb::InterruptCallback::instance.reset(new PythonInterruptCallback());

%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(double, NPY_FLOAT64);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(char, NPY_INT8);
REV_SWIG_PTR(unsigned short, NPY_UINT16);
REV_SWIG_PTR(short, NPY_INT16);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned int, NPY_UINT32);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);

#endif
