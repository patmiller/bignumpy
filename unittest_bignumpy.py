import unittest
import numpy
import bignumpy

class TestBignumpy(unittest.TestCase):
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

    def test_badshape(self):
        z = bignumpy.bignumpy('large0.raw','i',None)
        return
        z = bignumpy.bignumpy('large0.raw','i',None)
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',object())
        with self.assertRaises(RuntimeError):
            z = bignumpy.bignumpy('large0.raw','i',(1,2,3,None))
        return

    def test_intshape(self):
        z = bignumpy.bignumpy('large1.raw','i',7)
        return

    def test_create(self):
        z = bignumpy.bignumpy('large2.raw','i',(10,10))
        print 'z is',z
        assert isinstance(z,numpy.ndarray),'expected array, got %r'%type(z)
        return

if __name__ == "__main__":
    unittest.main()
