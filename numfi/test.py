import unittest
import numpy as np
from numfi import *
# TODO: add more test
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

        x = numfi([1,2,3],1,16,8)[0]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = numfi(np.zeros((3,3)),1,16,8)[2,1:3]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = np.arange(10).view(numfi)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 11)

        x = numfi(np.pi)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 13)

        x = numfi()
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 15)

        x = numfi([0,0,0])
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 15)

        x = numfi([0,1,0])
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 14)

        x = numfi(1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 4)

        x = numfi(-1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 5)

        x = numfi(102400)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, -2)

        x = numfi(0.1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 18)

        self.assertRaises(AssertionError, lambda:numfi(np.pi,1,0,0)) # numfi.w <=0

    def test_negtive_f_i(self):
        x = numfi(0.0547,0,8,10)
        self.assertEqual(x.f,10)
        self.assertEqual(x.i,-2)
        self.assertEqual(x.int,56)
        self.assertEqual(x.upper,0.249023437500000)
        self.assertEqual(x.lower,0)

        y = numfi(547,1,8,-3)
        self.assertEqual(y.f,-3)
        self.assertEqual(y.i,10)
        self.assertEqual(y.int,68)
        self.assertEqual(y.upper,1016)
        self.assertEqual(y.lower,-1024)

    def test_like(self):
        T = numfi([],1,17,5, RoundingMethod='Floor', OverflowAction='Wrap', FullPrecision=True)
        x = numfi([1,2,3,4], like=T)
        self.assertEqual(x.s, T.s)
        self.assertEqual(x.w, T.w)
        self.assertEqual(x.f, T.f)        
        self.assertEqual(x.RoundingMethod, T.RoundingMethod)
        self.assertEqual(x.OverflowAction, T.OverflowAction)
        self.assertEqual(x.FullPrecision, T.FullPrecision)
        # test us input_array as like 
        y = numfi(x)
        self.assertEqual(y.s, T.s)
        self.assertEqual(y.w, T.w)
        self.assertEqual(y.f, T.f)        
        self.assertEqual(y.RoundingMethod, T.RoundingMethod)
        self.assertEqual(y.OverflowAction, T.OverflowAction)
        self.assertEqual(y.FullPrecision, T.FullPrecision)

    def test_kwargs(self):
        x = numfi(np.pi,1,16,8,RoundingMethod='Floor',OverflowAction='Wrap',FullPrecision=True)
        self.assertEqual(x.RoundingMethod, 'Floor')
        self.assertEqual(x.OverflowAction, 'Wrap')
        self.assertEqual(x.FullPrecision, True)
        # test priority
        y = numfi(np.arange(10),0,22,like=x)
        self.assertEqual(y.s, 0)
        self.assertEqual(y.w, 22)
        self.assertEqual(y.f, x.f)
        self.assertEqual(y.RoundingMethod, x.RoundingMethod)
        self.assertEqual(y.OverflowAction, x.OverflowAction)
        self.assertEqual(y.FullPrecision, x.FullPrecision)

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
        x = numfi(0.0123456,1, 5, 8)
        self.assertEqual(x, 0.011718750000000)
        self.assertEqual(x.bin, '00011')

    def test_int64_overflow(self):
        self.assertRaises(OverflowError, lambda: numfi([1],1,65,64))

    def test_rounding(self):
        x = numfi(-3.785,1,14,6, RoundingMethod='Nearest')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

        x = numfi(-3.785,1,14,6, RoundingMethod='Floor')
        self.assertEqual(x, -3.796875000000000)
        self.assertEqual(x.bin, '11111100001101')

        x = numfi(-3.785,1,14,6, RoundingMethod='Ceiling')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

        x = numfi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

    def test_overflow(self):
        x = numfi([-4,-3,-2,-1,0,1,2,3],1,10,8,OverflowAction='Saturate')
        self.assertTrue(np.all(x==[-2.,-2.,-2.,-1.,0,1.,1.996093750000000,1.996093750000000]))
        x = numfi([-4,-3,-2,-1,0,1,2,3],0,10,8,OverflowAction='Saturate')
        self.assertTrue(np.all(x==[0,0,0,0,0,1,2,3]))
        x = numfi([1,2,3],1,10,8,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[1,-2,-1]))
        x = numfi([-4+4/11,-3+3/11,-2+2/11,-1+1/11,0,1+1/11,2+2/11,3+3/11],1,10,8,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[0.363281250000000,1.273437500000000,-1.816406250000000,-0.910156250000000,0,1.089843750000000,-1.816406250000000,-0.726562500000000]))
        x = numfi([1.1,2.2,3.3,4.4,5.5],0,6,4,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[1.125000000000000,2.187500000000000,3.312500000000000,0.375000000000000,1.500000000000000]))

    def test_bin_hex(self):
        x = numfi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x.bin, '11111100001110')
        self.assertEqual(x.bin_, '11111100.001110')
        self.assertEqual(x.hex, '-F2')
        
        y = numfi(547,1,8,-3)
        self.assertEqual(y.bin,  '01000100')
        self.assertEqual(y.bin_, '01000100xxx.')

        z = numfi(0.0547,0,8,10)
        self.assertEqual(z.bin,  '00111000')
        self.assertEqual(z.bin_, '.xx00111000')

    def test_i(self):
        x = numfi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x.i, 7)

        x = numfi(3.785,0,22,14, RoundingMethod='Zero')
        self.assertEqual(x.i, 8)
    
    def test_upper_lower_precision(self):
        x = numfi([],1,15,6)
        self.assertEqual(x.upper, 255.984375)
        self.assertEqual(x.lower, -256)
        self.assertEqual(x.precision, 0.015625)

        x = numfi([],0,15,6)
        self.assertEqual(x.upper,511.984375)
        self.assertEqual(x.lower,0)
        self.assertEqual(x.precision, 0.015625)

    def test_requantize(self):
        a = 1.99999999
        s168 = numfi(a, 1, 16, 8)
        s3216 = numfi(a, 1, 32, 16)
        s84 = numfi(a, 1, 8, 4)

        x = numfi(s168)
        y = numfi(x, 1, 32, 16)
        z = numfi(x, 1 ,8, 4)
        w = numfi(x, 1, 32, 4)
        u = numfi(x, 1, 32, 28)

        self.assertAlmostEqual(x, s168)
        self.assertAlmostEqual(y, x)
        self.assertAlmostEqual(z, s84)
        self.assertAlmostEqual(w, s84)
        self.assertAlmostEqual(u, x)

    def test_setitem(self):
        a = 1.99999999
        s168 = numfi(a, 1, 16, 8)
        s3216 = numfi(a, 1, 32, 16)
        s84 = numfi(a, 1, 8, 4)

        x = numfi(s168)
        x[0] = s3216[0]
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertAlmostEqual(x, s168)

        x[0] = s84[0]
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertAlmostEqual(x, s84)


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

        x += np.int64([256])
        self.assertTrue(np.all(x==[128.99609375, 129.99609375, 130.99609375, 131.99609375]))
        self.assertEqual(x.w, 17)
        self.assertEqual(x.f, 8)

        z = x + numfi(np.pi,0,14,11)
        self.assertEqual(z.s, 1)
        self.assertEqual(z.w, 22)
        self.assertEqual(z.f, 11)

        q = x + numfi(np.pi,1,14,11)
        self.assertEqual(q.s, 1)
        self.assertEqual(q.w, 21)
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

    def test_not_FullPrecision(self):
        x = numfi([1,2,3,4],1,16,8,FullPrecision=False)
        y = numfi([2,3,4,5],1,17,9,FullPrecision=False)
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

        xy = x*y
        self.assertEqual(xy.s, x.s)
        self.assertEqual(xy.w, x.w)
        self.assertEqual(xy.f, x.f)

        y1 = 1/y
        self.assertEqual(y1.s, y.s)
        self.assertEqual(y1.w, y.w)
        self.assertEqual(y1.f, y.f)

    def test_mul(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8)
        a3 = a * 3
        self.assertTrue(np.all(a3==[2.449218750000000 , 2.718750000000000, 0.386718750000000]))
        self.assertEqual(a3.s, 1)
        self.assertEqual(a3.w, 32)
        self.assertEqual(a3.f, 21) # note this is different than matlab

        aa = a*numfi(q,1,8,4)
        self.assertTrue(np.all(aa==[0.663330078125000,   0.792968750000000,   0.016113281250000]))
        self.assertEqual(aa.w, 24)
        self.assertEqual(aa.f, 12)

    def test_div(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8)
        a3 = a / 0.3333
        self.assertTrue(np.all(a3==[2.449218750000000 ,  2.718750000000000 ,  0.386718750000000]))
        self.assertEqual(a3.s, 1)
        self.assertEqual(a3.w, 16)
        self.assertEqual(a3.f, 8)

        aa = a/numfi(q,1,8,4)
        self.assertTrue(np.all(aa==[1.000000000000000 ,  1.062500000000000 ,  1.062500000000000]))
        self.assertEqual(aa.w, 16)
        self.assertEqual(aa.f, 4)

    def test_iop(self):
        x = numfi(1.12345,1,16,7)
        x += 0.5231
        self.assertEqual(x,1.648437500000000)
        self.assertEqual(x.w,17)
        self.assertEqual(x.f,7)

    def test_fixed_M(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = numfi(q,1,16,8,FullPrecision=True)
        a3 = a / 0.3333
        self.assertTrue(np.all(a3==[2.449218750000000 ,  2.718750000000000 ,  0.386718750000000]))
        self.assertEqual(a3.w,16)
        self.assertEqual(a3.f,8)
        
    def test_neg(self):
        x = numfi([1,2,3],1,16,8)
        self.assertTrue(np.all(-x==[-1,-2,-3]))
        self.assertEqual(x.s,1)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

        x = numfi([1,2,3],0,16,8)
        self.assertTrue(np.all(-x==[0,0,0]))
        self.assertEqual(x.s,0)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

    def test_invert(self):
        x = numfi([1,2,3],1,16,8)
        self.assertTrue(np.all(~x==[-1.00390625, -2.00390625, -3.00390625]))
        self.assertEqual(x.s,1)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

        x = numfi([1,2,3],0,16,8)
        self.assertTrue(np.all(~x==[0,0,0]))
        self.assertEqual(x.s,0)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

    def test_pow(self):
        x = numfi([0,1+1/77,-3-52/123],1,16,8)
        y = x**3
        self.assertTrue(np.all(y==[  0.        ,   1.03515625, -40.06640625]))
        self.assertEqual(y.s,1)
        self.assertEqual(y.w,16)
        self.assertEqual(y.f,8)
    
    def test_bitwise(self):
        n = np.array([0b1101,0b1001,0b0001,0b1111])/2**8
        x = numfi(n,1,16,8)
        self.assertTrue(np.all((x & 0b1100).int==[0b1100,0b1000,0b0000,0b1100]))
        self.assertTrue(np.all((x | 0b0101).int==[0b1101,0b1101,0b0101,0b1111]))
        self.assertTrue(np.all((x ^ 0b0110).int==[0b1011,0b1111,0b0111,0b1001]))
        self.assertTrue(np.all((x >>     1).int==[0b0110,0b0100,0b0000,0b0111]))
        self.assertTrue(np.all((x <<     1).int==[0b11010,0b10010,0b00010,0b11110]))

    def test_logical(self):
        x = numfi([-2,-1,0,1,2])
        self.assertTrue(np.all((x>1)==[False,False,False,False,True]))
        self.assertTrue(np.all((x>=1)==[False,False,False,True,True]))
        self.assertTrue(np.all((x==1)==[False,False,False,True,False]))
        self.assertTrue(np.all((x!=1)==[True,True,True,False,True]))
        self.assertTrue(np.all((x<=1)==[True,True,True,True,False]))
        self.assertTrue(np.all((x<1)==[True,True,True,False,False]))

    def test_ufunc(self):
        x = np.array([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])/100
        y = numfi(x,1,16,9)
        z = y[0]
        a = np.cos(x)
        b = np.cos(y)
        c = np.cos(y.double)
        d = numfi(c)
        self.assertTrue(np.all(b.int==d.int))
        self.assertTrue(np.all(b.int==[32767,   32761,   32750,   32731,   32701]))
        e = np.arctan2(b,y)
        self.assertTrue(np.all(e.int==[12867,   12739,   12611,   12483,   12355]))
        self.assertEqual(e.f,13)

if __name__ == '__main__':
    unittest.main()