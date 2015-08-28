/*
The MIT License (MIT)

Copyright (c) 2015 Pat Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"

#include "numpy/ndarrayobject.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <sys/mman.h>

struct BignumpyObject {
  char ignore[NPY_SIZEOF_PYARRAYOBJECT];
  int fd;
  void* map;
  size_t size;
};

static PyObject* NUMPY = NULL;
static PyObject* EMPTY = NULL;
static PyObject* ZEROS = NULL;

static PyTypeObject Bignumpy_Type;

static PyObject* bignumpy(PyObject* self, PyObject* args) {
  const char* filename = NULL;
  PyObject* dtype = NULL;
  PyObject* shape = Py_None;
  int fd;
  PyArrayObject* z;
  PyArray_Descr* descr;
  int nd = 0;
#define BIGNUMPY_MAXDIMS (128)
  npy_intp dims[BIGNUMPY_MAXDIMS];
  npy_intp strides[BIGNUMPY_MAXDIMS];
  size_t nelm;
  int i;
  npy_intp stride;
  PyObject* obj;
  struct BignumpyObject* bno;

  if (!PyArg_ParseTuple(args,"sO|O",&filename,&dtype,&shape)) return NULL;

  // We actually start with the type.  If we can't figure out the size
  // it is useless to open the file.  dtype can be almost anything, so
  // here, we create an array of zeros to get information about the
  // type using the actual numpy interface
  z = /*owned*/ (PyArrayObject*)PyObject_CallFunctionObjArgs(ZEROS,EMPTY,dtype,NULL);
  if (!z) return NULL;
  Py_INCREF(descr = PyArray_DESCR(z));
  Py_DECREF(z); z = NULL;

  // OK, we can open the file.  If it does not exist, we may have to create it
  fd = open(filename,O_RDWR|O_CREAT,0666);
  if (fd < 0) {
    Py_DECREF(descr);
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename);
  }


  // Figure out the current size of the file.
  struct stat status;
  if (fstat(fd,&status) < 0) {
    Py_DECREF(descr);
    close(fd);
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename);
  }

  //If the size is zero and we have a shape,
  // then we'll use ftruncate to change the size.  If we have no shape,
  // assume shape is (size(file)/elsize,)
  if (shape == Py_None) {
    strides[nd] = descr->elsize;
    dims[nd++] = status.st_size/descr->elsize;
  } else {
    PyObject* iterator = PyObject_GetIter(shape);
    if (!iterator) {
      long v = PyInt_AsLong(shape);
      if (v == -1 && PyErr_Occurred()) {
	PyErr_SetString(PyExc_RuntimeError,"invalid shape");
	Py_DECREF(descr);
	close(fd);
	return NULL;
      }
      dims[nd++] = v;
    } else {
      PyObject* item;
      while((item = PyIter_Next(iterator))) {
	if (nd >= BIGNUMPY_MAXDIMS) {
	  Py_DECREF(iterator);
	  Py_DECREF(item);
	  Py_DECREF(descr);
	  close(fd);
	  PyErr_SetString(PyExc_RuntimeError,"shape has too many dimensions");
	  return NULL;
	}
	long v = PyInt_AsLong(item);
	if (v == -1 && PyErr_Occurred()) {
	  Py_DECREF(iterator);
	  Py_DECREF(item);
	  Py_DECREF(descr);
	  close(fd);
	  PyErr_SetString(PyExc_RuntimeError,"invalid shape");
	  return NULL;
	}
	strides[nd] = 1;
	dims[nd++] = v;
	Py_DECREF(item);
      }
      Py_DECREF(iterator);
    }
  }

  // ----------------------------------------------------------------------
  // Compute the number of required elements
  // ----------------------------------------------------------------------
  nelm = 0;
  if (nd > 0) {
    nelm = 1;
    for(i=0;i<nd;++i) nelm *= dims[i];
  }

  // ---------------------------------------------------------------------- 
  // The strides include the element size.  We compute from back to front
  // ----------------------------------------------------------------------
  stride = descr->elsize;
  for(i=0;i<nd;++i) {
    strides[nd-1-i] = stride;
    stride *= dims[nd-1-i];
  }

  // ----------------------------------------------------------------------
  // Grow (but do not shrink) to be the expected size
  // ----------------------------------------------------------------------
  off_t expected = nelm * descr->elsize;
  if (status.st_size < expected) {
    if (ftruncate(fd,expected) < 0) {
      Py_DECREF(descr);
      close(fd);
      return PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename);
    }
  }

  // ----------------------------------------------------------------------
  // At this point, we can map the values into memory
  // ----------------------------------------------------------------------
  if (!expected) expected = 1; // mmap doesn't like 0 size
  void* m = mmap(NULL,expected,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if (m == MAP_FAILED) {
    Py_DECREF(descr);
    close(fd);
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename);
  }

  // ----------------------------------------------------------------------
  // Make a C object to hold the map state (pointer and size);
  // ----------------------------------------------------------------------

  obj = PyArray_NewFromDescr(
			      &Bignumpy_Type,
			      descr, /* steals reference */
			      nd,
			      dims,
			      strides,
			      m,
			      NPY_ARRAY_DEFAULT,
			      NULL);
  bno = (struct BignumpyObject*)obj;

  bno->fd = fd;
  bno->map = m;
  bno->size = expected;
  return obj;
}

static PyMethodDef methods[] = {
  {"bignumpy",bignumpy,METH_VARARGS,"doc"},
  {NULL}
};

void bignumpy_dealloc(PyObject* obj) {
  struct BignumpyObject* bno = (struct BignumpyObject*)obj;

  if (bno->map) {
    munmap(bno->map,bno->size);
    close(bno->fd);
    PyArray_Type.tp_dealloc(obj);
  }
}

PyObject* bignumpy_alloc(PyTypeObject *type, Py_ssize_t items) {
  PyObject* obj = type->tp_base->tp_alloc(type,items);
  struct BignumpyObject* bno = (struct BignumpyObject*)obj;
  bno->fd = 0;
  bno->map = NULL;
  bno->size = 0;
  return obj;
}

PyMODINIT_FUNC
initbignumpy(void) {
  PyObject* m = Py_InitModule("bignumpy",methods);
  if (m == NULL) return;

  // Initialize numpy API jump vectors
  import_array();

  Bignumpy_Type.tp_name = "bignumpy.Bignumpy";
  Bignumpy_Type.tp_dealloc = bignumpy_dealloc;
  Bignumpy_Type.tp_basicsize = sizeof(struct BignumpyObject);
  Bignumpy_Type.tp_alloc = bignumpy_alloc;
  Bignumpy_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  Bignumpy_Type.tp_base = &PyArray_Type;
  if (PyType_Ready(&Bignumpy_Type) < 0) return;
  Py_INCREF(&Bignumpy_Type);
  PyModule_AddObject(m, "Bignumpy", (PyObject*)&Bignumpy_Type);

  PyModule_AddIntConstant(m,"MAXDIM",32);

  // We need the actual numpy.zeros for our API to convert dtypes
  NUMPY = PyImport_ImportModule("numpy");
  if (!NUMPY) return;

  ZEROS = PyObject_GetAttrString(NUMPY,"zeros");
  if (!ZEROS) return;

  EMPTY = PyTuple_New(0);
  if (!EMPTY) return;
}
