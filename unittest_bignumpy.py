import unittest
import numpy
import bignumpy
import struct

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

    def xtest_maxdim(self):
        z = bignumpy.bignumpy('large1.raw','i',[1]*bignumpy.MAXDIM)
        return

    def xtest_badshape(self):
        z = bignumpy.bignumpy('large0.raw','i',None)
        return
        z = bignumpy.bignumpy('large0.raw','i',None)
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',object())
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',(1,2,3,None))
        return

    def xtest_intshape(self):
        z = bignumpy.bignumpy('large1.raw','i',7)
        return

    def xtest_create(self):
        z = bignumpy.bignumpy('large2.raw','i',(10,10))
        print 'z is',z
        assert isinstance(z,numpy.ndarray),'expected array, got %r'%type(z)
        return

if __name__ == "__main__":
    unittest.main()
