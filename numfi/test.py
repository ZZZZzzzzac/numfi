import unittest
import numpy as np
from .numfi import *

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
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertRaises(ValueError, lambda:numfi(np.pi,1,8,9))

        x = numfi([1,2,3],1,16,8)[0]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = numfi(np.zeros((3,3)),1,16,8)[2,1:3]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

    def test_like(self):
        T = numfi([],1,17,5, rounding='floor', overflow='wrap', fixed=True)
        x = numfi([1,2,3,4], like=T)
        self.assertEqual(x.w, T.w)
        self.assertEqual(x.f, T.f)
        self.assertEqual(x.s, T.s)
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
        self.assertEqual(z.s, 1)
        self.assertEqual(z.w, 20)
        self.assertEqual(z.f, 11)

        q = x + numfi(np.pi,1,14,11)
        self.assertEqual(q.s, 1)
        self.assertEqual(q.w, 20)
        self.assertEqual(q.f, 11)

    def test_sub(self):
        x = numfi([1,2,3,4],1,16,8) - 3
        self.assertTrue(np.all(x==[-2,-1,0,1]))
        self.assertEqual(x.w, 17)
        self.assertEqual(x.f, 8)

        y = 3 - numfi([1,2,3,4],1,16,8)
        self.assertTrue(np.all(y==[2,1,0,-1]))
        self.assertEqual(y.s, 1)
        self.assertEqual(y.w, 17)
        self.assertEqual(y.f, 8)

    def test_fixed_A(self):
        x = numfi([1,2,3,4],1,16,8,fixed=True)
        y = numfi([2,3,4,5],1,17,9,fixed=True)
        z = numfi(0,1,12,4)

        x1 = x+1
        self.assertEqual(x1.s, x.s)
        self.assertEqual(x1.w, x.w)
        self.assertEqual(x1.f, x.f)

        xy = x+y
        self.assertEqual(xy.s, x.s)
        self.assertEqual(xy.w, x.w)
        self.assertEqual(xy.f, x.f)

        y1 = 1+y
        self.assertEqual(y1.s, y.s)
        self.assertEqual(y1.w, y.w)
        self.assertEqual(y1.f, y.f)

        xz = x-z 
        self.assertEqual(xz.s, x.s)
        self.assertEqual(xz.w, x.w)
        self.assertEqual(xz.f, x.f)

    def test_mul(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8)
        a3 = a * 3
        self.assertTrue(np.all(a3==[2.449218750000000 , 2.718750000000000, 0.386718750000000]))
        self.assertEqual(a3.s, 1)
        self.assertEqual(a3.w, 32)
        self.assertEqual(a3.f, 16) # note this is different than matlab

        aa = a*numfi(q,1,8,4)
        self.assertEqual(aa.w, 24)
        self.assertEqual(aa.f, 12)

    def test_div(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8)
        a3 = a / 0.3333        
        self.assertEqual(a3.s, 1)
        self.assertEqual(a3.w, 32)
        self.assertEqual(a3.f, 16) # note this is different than matlab

        aa = a/numfi(q,1,8,4)
        self.assertEqual(aa.w, 24)
        self.assertEqual(aa.f, 12)

    def test_fixed_M(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8,fixed=True)
        a3 = a / 0.3333
        self.assertTrue(np.all(a3==[ 2.457031250000000  , 2.730468750000000  , 0.386718750000000]))

    # def test_i(self):
    #     x = numfi([1,2,3],1,15,6)
    #     before = x.ctypes.data
    #     x += 1
    #     after = x.ctypes.data
    #     self.assertEqual(before, after)
    #     x -= 2
    #     after = x.ctypes.data
    #     self.assertEqual(before, after)
    #     x *= 3
    #     after = x.ctypes.data
    #     self.assertEqual(before, after)
    #     x /= 4
    #     after = x.ctypes.data
    #     self.assertEqual(before, after)
        
    def test_neg(self):
        x = numfi([1,2,3],1,16,8)
        self.assertTrue(np.all(-x==[-1,-2,-3]))

        x = numfi([1,2,3],0,16,8)
        self.assertTrue(np.all(-x==[0,0,0]))