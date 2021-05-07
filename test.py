import unittest
from numfi import np,numfi

class numfiTest(unittest.TestCase):
    def test_create_numfi(self):        
        numfi(np.pi)
        numfi([np.pi])
        numfi([np.pi,-np.pi])
        numfi(np.array([np.pi,np.pi]))
        numfi(np.float32(np.pi))
        numfi(666)
    def test_swf(self):
        x = numfi(np.pi,1,16,8)
        self.assertEqual(x.signed, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertRaises(ValueError, lambda:numfi(np.pi,1,8,9))
    def test_kwargs(self):
        x = numfi(np.pi,1,16,8,rounding='floor',overflow='wrap',fixed=True)
        self.assertEqual(x.rounding, 'floor')
        self.assertEqual(x.overflow, 'wrap')
        self.assertEqual(x.fixed, True)
    def test_quantize(self):
        x = numfi(np.pi,1,16,8)
        self.assertEqual(x, 3.140625000000000)
        self.assertEqual(x.bin, '0000001100100100')        
        x = numfi(np.pi,0,8,4)
        self.assertEqual(x, 3.125000000000000)
        self.assertEqual(x.bin, '00110010')
        x = numfi(1.234567890,1,14,11)
        self.assertEqual(x, 1.234375000000000)
        self.assertEqual(x.bin, '00100111100000')
        x = numfi(-3.785,1,14,6)
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')
        x = numfi(-3.785,1,14,6, rounding='floor')
        self.assertEqual(x, -3.796875000000000)
        self.assertEqual(x.bin, '11111100001101')
    def test_overflow(self):
        x = numfi([1,2,3],1,10,8,overflow='saturate')
        self.assertTrue(np.all(x==[1,1.996093750000000,1.996093750000000]))
        x = numfi([1,2,3],1,10,8,overflow='wrap')
        self.assertTrue(np.all(x==[1,-2,-1]))
        x = numfi([1,2+1/11,3+3/11],1,10,8,overflow='wrap')
        self.assertTrue(np.all(x==[1.000000000000000,-1.910156250000000,-0.726562500000000]))


        
        