import numpy
import os
from distutils.core import setup,Extension

NUMPY=os.path.split(numpy.__file__)[0]

setup(name='bignumpy',
      version='0.1',
      description='File backed arrays for numpy',
      author='Pat Miller',
      author_email='patrick.miller@gmail.com',
      url='https://github.com/patmiller/bignumpy.git',
      ext_modules=[
        Extension('bignumpy',
                  ['bignumpy.c']),
        ]
     )

