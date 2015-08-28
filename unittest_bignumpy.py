# ------------------------------------------------------------------------
# The MIT License (MIT)
# 
# Copyright (c) 2015 Pat Miller
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------

import unittest
import numpy
import bignumpy
import struct
import os

class TestBignumpy(unittest.TestCase):
    def test_read(self):
        # Make a file with 1..12 in int32
        open('sample.raw','w').write(struct.pack('12i',*range(1,13)))

        # If we read it without shape, we get a 1-D vector
        a = bignumpy.bignumpy('sample.raw',numpy.int32)
        assert len(a)==12,'The length should be 12 here'
        assert all(a == range(1,13))

        # We can assign a shape
        b = bignumpy.bignumpy('sample.raw',numpy.int32,(12,))
        assert len(b)==12,'The length should be 12 here'
        assert all(b == range(1,13))

        # We can assign a shape smaller than the real size
        c = bignumpy.bignumpy('sample.raw',numpy.int32,(8,))
        assert len(c)==8,'The length should be 8 here'
        assert all(c == range(1,9))


        # We can assign other shapes
        d = bignumpy.bignumpy('sample.raw',numpy.int32,(3,4))
        assert all((d == [[ 1,  2,  3,  4],
                          [ 5,  6,  7,  8],
                          [ 9, 10, 11, 12]]).flatten())

        e = bignumpy.bignumpy('sample.raw',numpy.int32,(3,2,2))
        assert all((e == [[[ 1,  2],
                           [ 3,  4]],
                          
                          [[ 5,  6],
                           [ 7,  8]],
                          
                          [[ 9, 10],
                           [11, 12]]]).flatten())


        # We can assign a shape larger than the real size
        # and it gets padded with 0
        f = bignumpy.bignumpy('sample.raw',numpy.int32,(16,))
        assert len(f)==16,'The length should be 16 here'
        assert all(f == range(1,13)+[0,0,0,0])

        return

    def test_baddescr(self):
        with self.assertRaises(TypeError):
            z = bignumpy.bignumpy('large0.raw','????')
        return

    def test_tomanydims(self):
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',[1]*(bignumpy.MAXDIM+1))
        return

    def test_maxdim(self):
        z = bignumpy.bignumpy('large1.raw','i',[1]*bignumpy.MAXDIM)
        return

    def test_defaultshapezero(self):
        try: os.unlink('large0.raw')
        except OSError: pass
        z = bignumpy.bignumpy('large0.raw','i',None)
        assert z.shape == (0,)
        return

    def test_badshape(self):
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',object())
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',(1,2,3,None))
        return

    def test_intshape(self):
        z = bignumpy.bignumpy('large1.raw','i',7)
        assert z.shape == (7,)
        return

    def test_create(self):
        try: os.unlink('large2.raw')
        except OSError: pass
        z = bignumpy.bignumpy('large2.raw','i',(10,10))
        assert isinstance(z,numpy.ndarray),'expected array, got %r'%type(z)
        assert z.shape == (10,10)
        assert all(z.flatten() == [0]*100)
        return

if __name__ == "__main__":
    unittest.main()
