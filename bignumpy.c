#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"

#include "numpy/ndarrayobject.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <sys/mman.h>

struct mapstate {
  void* m;
  size_t size;
  int fd;
};

void destroy(void* state) {
  free(state);
}

static PyObject* NUMPY = NULL;
static PyObject* EMPTY = NULL;
static PyObject* ZEROS = NULL;

static PyTypeObject BignumpyType;

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
    strides[nd] = 1;
    dims[nd++] = status.st_size;
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
      puts("Set dim");
      strides[nd] = 1;
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
  // Compute the number of required elements
  size_t nelm = 0;
  if (nd > 0) {
    nelm = 1;
    int i;
    for(i=0;i<nd;++i) nelm *= dims[i];
  }

  // ----------------------------------------------------------------------
  // Grow (but do not shrink) to be the expected size
  // ----------------------------------------------------------------------
  off_t expected = nelm * descr->elsize;
  printf("Expected %ld, have %ld\n",(long)expected,(long)status.st_size);
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
    printf("%s:%d: map fail here\n",__FILE__,__LINE__);
    Py_DECREF(descr);
    close(fd);
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename);
  }

  // ----------------------------------------------------------------------
  // Make a C object to hold the map state (pointer and size);
  // ----------------------------------------------------------------------
  struct mapstate* cobject = (struct mapstate*)malloc(sizeof(struct mapstate));
  cobject->m = m;
  cobject->size = expected;
  cobject->fd = fd;
  printf("Save m: %p sz: %ld fd: %d\n",m,expected,fd);
  PyObject* vv = PyCObject_FromVoidPtr(cobject,destroy);
  PyObject* zz = PyArray_NewFromDescr(
			      &BignumpyType,
			      descr, /* steals reference */
			      nd,
			      dims,
			      strides,
			      m,
			      NPY_ARRAY_DEFAULT,
			      vv);
  return zz;
}

static PyMethodDef methods[] = {
  {"bignumpy",bignumpy,METH_VARARGS,"doc"},
  {NULL}
};


static int initialize(PyObject* self, PyObject* args, PyObject* kw) {
  return -1;
}

static PyObject* __array_finalize__(PyObject* self, PyObject* object) {
  puts("in finalize");
  struct mapstate* cobject = (struct mapstate*)PyCObject_AsVoidPtr(object);
  printf("M: %p sz: %ld fd: %d\n",cobject->m,cobject->size,cobject->fd);
  munmap(cobject->m,cobject->size);
  close(cobject->fd);
  Py_DECREF(object);
  puts("done finalize");

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef bignumpy_methods[] = {
  {"__array_finalize__",__array_finalize__,METH_O,""},
  {NULL}
};

PyMODINIT_FUNC
initbignumpy(void) {
  PyObject* m = Py_InitModule("bignumpy",methods);
  if (m == NULL) return;

  // Initialize numpy API jump vectors
  import_array();

  BignumpyType.tp_name = "bignumpy.Bignumpy";
  BignumpyType.tp_basicsize = PyArray_Type.tp_basicsize;
  BignumpyType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  BignumpyType.tp_init = initialize;
  BignumpyType.tp_methods = bignumpy_methods;
  if (PyType_Ready(&BignumpyType) < 0) return;
  Py_INCREF(&BignumpyType);
  PyModule_AddObject(m, "Bignumpy", (PyObject*)&BignumpyType);

  PyModule_AddIntConstant(m,"MAXDIM",32);

  // We need the actual numpy.zeros for our API to convert dtypes
  NUMPY = PyImport_ImportModule("numpy");
  if (!NUMPY) return;

  ZEROS = PyObject_GetAttrString(NUMPY,"zeros");
  if (!ZEROS) return;

  EMPTY = PyTuple_New(0);
  if (!EMPTY) return;
}
