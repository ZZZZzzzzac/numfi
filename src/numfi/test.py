import unittest
from .numfi import np,numfi

class numfiTest(unittest.TestCase):
    def test_create_numfi(self):        
        numfi(np.pi)
        numfi([np.pi])
        numfi([np.pi,-np.pi])
        numfi(np.array([np.pi,np.pi]))
        numfi(np.float32(np.pi))
        numfi(666)
        numfi(numfi([1,2,3,4.5]))

    def test_swf(self):
        x = numfi(np.pi,1,16,8)
        self.assertEqual(x.signed, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertRaises(ValueError, lambda:numfi(np.pi,1,8,9))

    def test_like(self):
        T = numfi([],1,17,5, rounding='floor', overflow='wrap', fixed=True)
        x = numfi([1,2,3,4], like=T)
        self.assertEqual(x.w, T.w)
        self.assertEqual(x.f, T.f)
        self.assertEqual(x.signed, T.signed)
        self.assertEqual(x.rounding, T.rounding)
        self.assertEqual(x.overflow, T.overflow)
        self.assertEqual(x.fixed, T.fixed)

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
        x = numfi([-4,-3,-2,-1,0,1,2,3],1,10,8,overflow='saturate')
        self.assertTrue(np.all(x==[-2.,-2.,-2.,-1.,0,1.,1.996093750000000,1.996093750000000]))
        x = numfi([-4,-3,-2,-1,0,1,2,3],0,10,8,overflow='saturate')
        self.assertTrue(np.all(x==[0,0,0,0,0,1,2,3]))
        x = numfi([1,2,3],1,10,8,overflow='wrap')
        self.assertTrue(np.all(x==[1,-2,-1]))
        x = numfi([-4+4/11,-3+3/11,-2+2/11,-1+1/11,0,1+1/11,2+2/11,3+3/11],1,10,8,overflow='wrap')
        self.assertTrue(np.all(x==[0.363281250000000,1.273437500000000,-1.816406250000000,-0.910156250000000,0,1.089843750000000,-1.816406250000000,-0.726562500000000]))
        x = numfi([1.1,2.2,3.3,4.4,5.5],0,6,4,overflow='wrap')
        self.assertTrue(np.all(x==[1.125000000000000,2.187500000000000,3.312500000000000,0.375000000000000,1.500000000000000]))

    def test_add(self):
        x = numfi([1,2,3,4],1,16,8)
        x_plus_1 = x + 1
        self.assertTrue(np.all(x_plus_1==[2,3,4,5]))
        self.assertEqual(x_plus_1.w,17)
        self.assertEqual(x_plus_1.f,8)

        x_plus_y = x + numfi([1.5,2.5,3.5,4.5],1,16,8)
        self.assertTrue(np.all(x_plus_y==[2.5,4.5,6.5,8.5]))
        self.assertEqual(x_plus_y.w, 17)        
        self.assertEqual(x_plus_y.f, 8)

        x_plus_256 = x + np.int64([256])
        self.assertTrue(np.all(x_plus_256==[128.9960937500000, 129.9960937500000, 130.9960937500000, 131.9960937500000]))
        self.assertEqual(x_plus_256.w, 17)
        self.assertEqual(x_plus_256.f, 8)

        z = x + numfi(np.pi,0,14,11)
        self.assertEqual(z.signed, 1)
        self.assertEqual(z.w, 20)
        self.assertEqual(z.f, 11)

        q = x + numfi(np.pi,1,14,11)
        self.assertEqual(q.signed, 1)
        self.assertEqual(q.w, 20)
        self.assertEqual(q.f, 11)

    def test_sub(self):
        x = numfi([1,2,3,4],1,16,8) - 3
        self.assertTrue(np.all(x==[-2,-1,0,1]))
        self.assertEqual(x.w, 17)
        self.assertEqual(x.f, 8)

        
        