import unittest
import numpy as np
from numfi import numfi as fi   # change numfi to numqi to test numqi
# TODO: add more test
pi = np.pi
class fiTest(unittest.TestCase):
    def test_create_fi(self):        
        fi(pi)
        fi([pi])
        fi([pi,-pi])
        fi(np.array([pi,pi]))
        fi(np.float32(pi))
        fi(666)
        fi(fi([1,2,3,4.5]))

    def test_swf(self):
        x = fi(pi,1,16,8)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = fi([1,2,3],1,16,8)[0]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = fi(np.zeros((3,3)),1,16,8)[2,1:3]
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)

        x = np.arange(10).view(fi)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 11)

        x = fi(pi)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 13)

        x = fi()
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 15)

        x = fi([0,0,0])
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 15)

        x = fi([0,1,0])
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 14)

        x = fi(1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 4)

        x = fi(-1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 5)

        x = fi(102400)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, -2)

        x = fi(0.1024)
        self.assertEqual(x.s, 1)
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 18)

        self.assertRaises(AssertionError, lambda:fi(pi,1,0,0)) # fi.w <=0

    def test_negtive_f_i(self):
        x = fi(0.0547,0,8,10)
        self.assertEqual(x.f,10)
        self.assertEqual(x.i,-2)
        self.assertEqual(x.int,56)
        self.assertEqual(x.upper,0.249023437500000)
        self.assertEqual(x.lower,0)

        y = fi(547,1,8,-3)
        self.assertEqual(y.f,-3)
        self.assertEqual(y.i,10)
        self.assertEqual(y.int,68)
        self.assertEqual(y.upper,1016)
        self.assertEqual(y.lower,-1024)

    def test_like(self):
        T = fi([],1,17,5, RoundingMethod='Floor', OverflowAction='Wrap', FullPrecision=True)
        x = fi([1,2,3,4], like=T)
        self.assertEqual(x.s, T.s)
        self.assertEqual(x.w, T.w)
        self.assertEqual(x.f, T.f)        
        self.assertEqual(x.RoundingMethod, T.RoundingMethod)
        self.assertEqual(x.OverflowAction, T.OverflowAction)
        self.assertEqual(x.FullPrecision, T.FullPrecision)
        # test us input_array as like 
        y = fi(x)
        self.assertEqual(y.s, T.s)
        self.assertEqual(y.w, T.w)
        self.assertEqual(y.f, T.f)        
        self.assertEqual(y.RoundingMethod, T.RoundingMethod)
        self.assertEqual(y.OverflowAction, T.OverflowAction)
        self.assertEqual(y.FullPrecision, T.FullPrecision)

    def test_kwargs(self):
        x = fi(pi,1,16,8,RoundingMethod='Floor',OverflowAction='Wrap',FullPrecision=True)
        self.assertEqual(x.RoundingMethod, 'Floor')
        self.assertEqual(x.OverflowAction, 'Wrap')
        self.assertEqual(x.FullPrecision, True)
        # test priority
        y = fi(np.arange(10),0,22,like=x)
        self.assertEqual(y.s, 0)
        self.assertEqual(y.w, 22)
        self.assertEqual(y.f, x.f)
        self.assertEqual(y.RoundingMethod, x.RoundingMethod)
        self.assertEqual(y.OverflowAction, x.OverflowAction)
        self.assertEqual(y.FullPrecision, x.FullPrecision)

    def test_quantize(self):
        x = fi(pi,1,16,8)
        self.assertEqual(x, 3.140625000000000)
        self.assertEqual(x.bin, '0000001100100100')        
        x = fi(pi,0,8,4)
        self.assertEqual(x, 3.125000000000000)
        self.assertEqual(x.bin, '00110010')
        x = fi(1.234567890,1,14,11)
        self.assertEqual(x, 1.234375000000000)
        self.assertEqual(x.bin, '00100111100000')
        x = fi(-3.785,1,14,6)
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')
        x = fi(0.0123456,1, 5, 8)
        self.assertEqual(x, 0.011718750000000)
        self.assertEqual(x.bin, '00011')

    def test_int64_overflow(self):
        self.assertRaises(OverflowError, lambda: fi([1],1,65,64))

    def test_rounding(self):
        x = fi(-3.785,1,14,6, RoundingMethod='Nearest')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

        x = fi(-3.785,1,14,6, RoundingMethod='Floor')
        self.assertEqual(x, -3.796875000000000)
        self.assertEqual(x.bin, '11111100001101')

        x = fi(-3.785,1,14,6, RoundingMethod='Ceiling')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

        x = fi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x, -3.781250000000000)
        self.assertEqual(x.bin, '11111100001110')

    def test_overflow(self):
        x = fi([-4,-3,-2,-1,0,1,2,3],1,10,8,OverflowAction='Saturate')
        self.assertTrue(np.all(x==[-2.,-2.,-2.,-1.,0,1.,1.996093750000000,1.996093750000000]))
        x = fi([-4,-3,-2,-1,0,1,2,3],0,10,8,OverflowAction='Saturate')
        self.assertTrue(np.all(x==[0,0,0,0,0,1,2,3]))
        x = fi([1,2,3],1,10,8,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[1,-2,-1]))
        x = fi([-4+4/11,-3+3/11,-2+2/11,-1+1/11,0,1+1/11,2+2/11,3+3/11],1,10,8,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[0.363281250000000,1.273437500000000,-1.816406250000000,-0.910156250000000,0,1.089843750000000,-1.816406250000000,-0.726562500000000]))
        x = fi([1.1,2.2,3.3,4.4,5.5],0,6,4,OverflowAction='Wrap')
        self.assertTrue(np.all(x==[1.125000000000000,2.187500000000000,3.312500000000000,0.375000000000000,1.500000000000000]))
        self.assertRaises(OverflowError, lambda: fi([1,2,3],1,10,8,OverflowAction='Error'))

    def test_bin_hex(self):
        x = fi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x.bin, '11111100001110')
        self.assertEqual(x.bin_, '11111100.001110')
        self.assertEqual(x.hex, '3F0E')
        
        y = fi(547,1,8,-3)
        self.assertEqual(y.bin,  '01000100')
        self.assertEqual(y.bin_, '01000100xxx.')

        z = fi(0.0547,0,8,10)
        self.assertEqual(z.bin,  '00111000')
        self.assertEqual(z.bin_, '.xx00111000')

    def test_i(self):
        x = fi(-3.785,1,14,6, RoundingMethod='Zero')
        self.assertEqual(x.i, 7)

        x = fi(3.785,0,22,14, RoundingMethod='Zero')
        self.assertEqual(x.i, 8)
    
    def test_upper_lower_precision(self):
        x = fi([],1,15,6)
        self.assertEqual(x.upper, 255.984375)
        self.assertEqual(x.lower, -256)
        self.assertEqual(x.precision, 0.015625)

        x = fi([],0,15,6)
        self.assertEqual(x.upper,511.984375)
        self.assertEqual(x.lower,0)
        self.assertEqual(x.precision, 0.015625)

    def test_requantize(self):
        a = 1.99999999
        s168 = fi(a, 1, 16, 8)
        s3216 = fi(a, 1, 32, 16)
        s84 = fi(a, 1, 8, 4)

        x = fi(s168)
        y = fi(x, 1, 32, 16)
        z = fi(x, 1 ,8, 4)
        w = fi(x, 1, 32, 4)
        u = fi(x, 1, 32, 28)

        self.assertAlmostEqual(x, s168)
        self.assertAlmostEqual(y, x)
        self.assertAlmostEqual(z, s84)
        self.assertAlmostEqual(w, s84)
        self.assertAlmostEqual(u, x)

    def test_setitem(self):
        a = [1.99, 2.00, 2.01]
        s168 = fi(a, 1, 16, 8)
        s3216 = fi(a, 1, 32, 16)
        s84 = fi(a, 1, 8, 4)

        x = fi(s168)
        x[0] = s3216[0]
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertTrue(np.allclose(x, s168))

        x[0:2] = s84[0:2]
        self.assertEqual(x.w, 16)
        self.assertEqual(x.f, 8)
        self.assertTrue(np.allclose(x[0:2], s84[0:2]))

    def test_getitem(self):
        a = fi(np.random.randn(3,3),1,14,6) # 2d
        b = a[0]
        self.assertTrue(isinstance(b, type(a)))
        self.assertTrue(b.shape == (3,)) # 1d
        self.assertEqual(b.s, a.s)
        self.assertEqual(b.w, a.w)
        self.assertEqual(b.f, a.f)
        self.assertEqual(b.RoundingMethod, a.RoundingMethod)
        self.assertEqual(b.OverflowAction, a.OverflowAction)
        self.assertTrue(np.all(b.int==a.int[0]))

        b = a[0,0] # scalar, but still (1,) numfi
        self.assertTrue(isinstance(b, type(a)))
        self.assertTrue(b.shape == (1,))
        self.assertEqual(b.s, a.s)
        self.assertEqual(b.w, a.w)
        self.assertEqual(b.f, a.f)
        self.assertEqual(b.RoundingMethod, a.RoundingMethod)
        self.assertEqual(b.OverflowAction, a.OverflowAction)
        self.assertTrue(np.all(b.int==a.int[0,0]))

        b = a[1:2]
        self.assertTrue(isinstance(b, type(a)))
        self.assertEqual(b.s, a.s)
        self.assertEqual(b.w, a.w)
        self.assertEqual(b.f, a.f)
        self.assertEqual(b.RoundingMethod, a.RoundingMethod)
        self.assertEqual(b.OverflowAction, a.OverflowAction)
        self.assertTrue(np.all(b.int==a.int[1:2]))

        b = a[1,0:2]
        self.assertTrue(isinstance(b, type(a)))
        self.assertEqual(b.s, a.s)
        self.assertEqual(b.w, a.w)
        self.assertEqual(b.f, a.f)
        self.assertEqual(b.RoundingMethod, a.RoundingMethod)
        self.assertEqual(b.OverflowAction, a.OverflowAction)
        self.assertTrue(np.all(b.int==a.int[1,0:2]))

    def test_add(self):
        def check(x, y, s, w, f, integer):
            z = x + y
            self.assertEqual(z.s, s)
            self.assertEqual(z.w, w)
            self.assertEqual(z.f, f)
            self.assertTrue(np.all(z.int==integer))
    
        x = fi(pi,1,16,8)
        check(x, fi(0.1,1,24,8),1,25,8,830)
        check(x, fi(0.1,1,24,4),1,29,8,836)
        check(x, fi(0.1,1,24,12),1,25,12,13274)
        check(x, fi(0.1,1,16,8),1,17,8,830)
        check(x, fi(0.1,1,16,4),1,21,8,836)
        check(x, fi(0.1,1,16,12),1,21,12,13274)
        check(x, fi(0.1,1,8,8),1,17,8,830)
        check(x, fi(0.1,1,8,4),1,17,8,836)
        check(x, fi(0.1,1,8,12),1,21,12,12991)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,1,24,8),1,26,8,830)
        check(x, fi(0.1,1,24,4),1,30,8,836)
        check(x, fi(0.1,1,24,12),1,26,12,13274)
        check(x, fi(0.1,1,16,8),1,18,8,830)
        check(x, fi(0.1,1,16,4),1,22,8,836)
        check(x, fi(0.1,1,16,12),1,22,12,13274)
        check(x, fi(0.1,1,8,8),1,18,8,830)
        check(x, fi(0.1,1,8,4),1,18,8,836)
        check(x, fi(0.1,1,8,12),1,22,12,12991)

        x = fi(pi,1,16,8)
        check(x, fi(0.1,0,24,8),1,26, 8,830)
        check(x, fi(0.1,0,24,4),1,30, 8,836)
        check(x, fi(0.1,0,24,12),1,26,12,13274)
        check(x, fi(0.1,0,16,8),1,18, 8,830)
        check(x, fi(0.1,0,16,4),1,22, 8,836)
        check(x, fi(0.1,0,16,12),1,22,12,13274)
        check(x, fi(0.1,0,8,8),1,18, 8,830)
        check(x, fi(0.1,0,8,4),1,18, 8,836)
        check(x, fi(0.1,0,8,12),1,22,12,13119)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,0,24,8),0,25, 8,830)
        check(x, fi(0.1,0,24,4),0,29, 8,836)
        check(x, fi(0.1,0,24,12),0,25,12,13274)
        check(x, fi(0.1,0,16,8),0,17, 8,830)
        check(x, fi(0.1,0,16,4),0,21, 8,836)
        check(x, fi(0.1,0,16,12),0,21,12,13274)
        check(x, fi(0.1,0,8,8),0,17, 8,830)
        check(x, fi(0.1,0,8,4),0,17, 8,836)
        check(x, fi(0.1,0,8,12),0,21,12,13119)

        x = fi(pi,1,16,8)
        check(x, 4, 1,17, 8,1828)
        check(x, [4], 1,17, 8,1828)
        check(x, np.int64(4), 1,17, 8,1828)
        check(x, np.array([4]), 1,17, 8,1828)

    def test_sub(self):
        def check(x, y, s, w, f, integer):
            z = x - y
            self.assertEqual(z.s, s)
            self.assertEqual(z.w, w)
            self.assertEqual(z.f, f)
            self.assertTrue(np.all(z.int==integer))
    
        x = fi(pi,1,16,8)
        check(x, fi(0.1,1,24,8),1,25, 8,778)
        check(x, fi(0.1,1,24,4),1,29, 8,772)
        check(x, fi(0.1,1,24,12),1,25,12,12454)
        check(x, fi(0.1,1,16,8),1,17, 8,778)
        check(x, fi(0.1,1,16,4),1,21, 8,772)
        check(x, fi(0.1,1,16,12),1,21,12,12454)
        check(x, fi(0.1,1,8,8),1,17, 8,778)
        check(x, fi(0.1,1,8,4),1,17, 8,772)
        check(x, fi(0.1,1,8,12),1,21,12,12737)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,1,24,8),1,26, 8,778)
        check(x, fi(0.1,1,24,4),1,30, 8,772)
        check(x, fi(0.1,1,24,12),1,26,12,12454)
        check(x, fi(0.1,1,16,8),1,18, 8,778)
        check(x, fi(0.1,1,16,4),1,22, 8,772)
        check(x, fi(0.1,1,16,12),1,22,12,12454)
        check(x, fi(0.1,1,8,8),1,18, 8,778)
        check(x, fi(0.1,1,8,4),1,18, 8,772)
        check(x, fi(0.1,1,8,12),1,22,12,12737)

        x = fi(pi,1,16,8)
        check(x, fi(0.1,0,24,8),1,26, 8,778)
        check(x, fi(0.1,0,24,4),1,30, 8,772)
        check(x, fi(0.1,0,24,12),1,26,12,12454)
        check(x, fi(0.1,0,16,8),1,18, 8,778)
        check(x, fi(0.1,0,16,4),1,22, 8,772)
        check(x, fi(0.1,0,16,12),1,22,12,12454)
        check(x, fi(0.1,0,8,8),1,18, 8,778)
        check(x, fi(0.1,0,8,4),1,18, 8,772)
        check(x, fi(0.1,0,8,12),1,22,12,12609)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,0,24,8),0,25, 8,778)
        check(x, fi(0.1,0,24,4),0,29, 8,772)
        check(x, fi(0.1,0,24,12),0,25,12,12454)
        check(x, fi(0.1,0,16,8),0,17, 8,778)
        check(x, fi(0.1,0,16,4),0,21, 8,772)
        check(x, fi(0.1,0,16,12),0,21,12,12454)
        check(x, fi(0.1,0,8,8),0,17, 8,778)
        check(x, fi(0.1,0,8,4),0,17, 8,772)
        check(x, fi(0.1,0,8,12),0,21,12,12609)

        x = fi(pi,1,16,8)
        check(x, 4, 1,17, 8,-220)
        check(x, [4], 1,17, 8,-220)
        check(x, np.int64(4), 1,17, 8,-220)
        check(x, np.array([4]), 1,17, 8,-220)

    def test_not_FullPrecision(self):
        x = fi([1,2,3,4],1,16,8,FullPrecision=False)
        y = fi([2,3,4,5],0,32,4,FullPrecision=False)
        z = fi(0,1,12,4)

        x1 = x+1
        self.assertEqual(x1.s, x.s)
        self.assertEqual(x1.w, x.w)
        self.assertEqual(x1.f, x.f)

        xy = x+y
        self.assertEqual(xy.s, max(x.s,y.s))
        self.assertEqual(xy.w, max(x.w,y.w))
        self.assertEqual(xy.f, max(x.f,y.f))

        y1 = 1+y
        self.assertEqual(y1.s, y.s)
        self.assertEqual(y1.w, y.w)
        self.assertEqual(y1.f, y.f)

        xz = x-z 
        self.assertEqual(xz.s, max(x.s,z.s))
        self.assertEqual(xz.w, max(x.w,z.w))
        self.assertEqual(xz.f, max(x.f,z.f))

        zx = z+x
        self.assertEqual(zx.s, max(x.s,z.s))
        self.assertEqual(zx.w, max(x.w,z.w))
        self.assertEqual(zx.f, max(x.f,z.f))

        xy = x*y
        self.assertEqual(xy.s, max(x.s,y.s))
        self.assertEqual(xy.w, max(x.w,y.w))
        self.assertEqual(xy.f, max(x.f,y.f))

        y1 = 1/y
        self.assertEqual(y1.s, y.s)
        self.assertEqual(y1.w, y.w)
        self.assertEqual(y1.f, y.f)

    def test_mul(self):
        def check(x, y, s, w, f, integer):
            z = x * y
            self.assertEqual(z.s, s)
            self.assertEqual(z.w, w)
            self.assertEqual(z.f, f)
            self.assertTrue(np.all(z.int==integer))

        x = fi(pi,1,16,8)
        check(x, fi(0.1,1,24,8),1,40,16,20904)
        check(x, fi(0.1,1,24,4),1,40,12,1608)
        check(x, fi(0.1,1,24,12),1,40,20,329640)
        check(x, fi(0.1,1,16,8),1,32,16,20904)
        check(x, fi(0.1,1,16,4),1,32,12,1608)
        check(x, fi(0.1,1,16,12),1,32,20,329640)
        check(x, fi(0.1,1,8,8),1,24,16,20904)
        check(x, fi(0.1,1,8,4),1,24,12,1608)
        check(x, fi(0.1,1,8,12),1,24,20,102108)
        

        x = fi(pi,1,16,8)
        check(x, fi(0.1,0,24,8),1,40,16,20904)
        check(x, fi(0.1,0,24,4),1,40,12,1608)
        check(x, fi(0.1,0,24,12),1,40,20,329640)
        check(x, fi(0.1,0,16,8),1,32,16,20904)
        check(x, fi(0.1,0,16,4),1,32,12,1608)
        check(x, fi(0.1,0,16,12),1,32,20,329640)
        check(x, fi(0.1,0,8,8),1,24,16,20904)
        check(x, fi(0.1,0,8,4),1,24,12,1608)
        check(x, fi(0.1,0,8,12),1,24,20,205020)


        x = fi(pi,0,16,8)
        check(x, fi(0.1,1,24,8),1,40,16,20904)
        check(x, fi(0.1,1,24,4),1,40,12,1608)
        check(x, fi(0.1,1,24,12),1,40,20,329640)
        check(x, fi(0.1,1,16,8),1,32,16,20904)
        check(x, fi(0.1,1,16,4),1,32,12,1608)
        check(x, fi(0.1,1,16,12),1,32,20,329640)
        check(x, fi(0.1,1,8,8),1,24,16,20904)
        check(x, fi(0.1,1,8,4),1,24,12,1608)
        check(x, fi(0.1,1,8,12),1,24,20,102108)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,0,24,8),0,40,16,20904)
        check(x, fi(0.1,0,24,4),0,40,12,1608)
        check(x, fi(0.1,0,24,12),0,40,20,329640)
        check(x, fi(0.1,0,16,8),0,32,16,20904)
        check(x, fi(0.1,0,16,4),0,32,12,1608)
        check(x, fi(0.1,0,16,12),0,32,20,329640)
        check(x, fi(0.1,0,8,8),0,24,16,20904)
        check(x, fi(0.1,0,8,4),0,24,12,1608)
        check(x, fi(0.1,0,8,12),0,24,20,205020)

        x = fi(pi, 1, 18, 4)
        check(x, 0.3, 1, 36, 22, 3932150)
        check(x, -0.3, 1, 36, 22, -3932150)

    def test_div(self):
        def check(x, y, s, w, f, integer):
            z = x / y
            self.assertEqual(z.s, s)
            self.assertEqual(z.w, w)
            self.assertEqual(z.f, f)
            self.assertTrue(np.all(z.int==integer))

        x = fi(pi,1,16,8)
        check(x, fi(0.1,1,24,8),1,24, 0,31)
        check(x, fi(0.1,1,24,4),1,24, 4,402)
        check(x, fi(0.1,1,24,12),1,24,-4,2)
        check(x, fi(0.1,1,16,8),1,16, 0,31)
        check(x, fi(0.1,1,16,4),1,16, 4,402)
        check(x, fi(0.1,1,16,12),1,16,-4,2)
        check(x, fi(0.1,1,8,8),1,16, 0,31)
        check(x, fi(0.1,1,8,4),1,16, 4,402)
        check(x, fi(0.1,1,8,12),1,16,-4,6)

        x = fi(pi,1,16,8)
        check(x, fi(0.1,0,24,8),1,24, 0,31)
        check(x, fi(0.1,0,24,4),1,24, 4,402)
        check(x, fi(0.1,0,24,12),1,24,-4,2)
        check(x, fi(0.1,0,16,8),1,16, 0,31)
        check(x, fi(0.1,0,16,4),1,16, 4,402)
        check(x, fi(0.1,0,16,12),1,16,-4,2)
        check(x, fi(0.1,0,8,8),1,16, 0,31)
        check(x, fi(0.1,0,8,4),1,16, 4,402)
        check(x, fi(0.1,0,8,12),1,16,-4,3)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,1,24,8),1,24, 0,31)
        check(x, fi(0.1,1,24,4),1,24, 4,402)
        check(x, fi(0.1,1,24,12),1,24,-4,2)
        check(x, fi(0.1,1,16,8),1,16, 0,31)
        check(x, fi(0.1,1,16,4),1,16, 4,402)
        check(x, fi(0.1,1,16,12),1,16,-4,2)
        check(x, fi(0.1,1,8,8),1,16, 0,31)
        check(x, fi(0.1,1,8,4),1,16, 4,402)
        check(x, fi(0.1,1,8,12),1,16,-4,6)

        x = fi(pi,0,16,8)
        check(x, fi(0.1,0,24,8),0,24, 0,31)
        check(x, fi(0.1,0,24,4),0,24, 4,402)
        check(x, fi(0.1,0,24,12),0,24,-4,2)
        check(x, fi(0.1,0,16,8),0,16, 0,31)
        check(x, fi(0.1,0,16,4),0,16, 4,402)
        check(x, fi(0.1,0,16,12),0,16,-4,2)
        check(x, fi(0.1,0,8,8),0,16, 0,31)
        check(x, fi(0.1,0,8,4),0,16, 4,402)
        check(x, fi(0.1,0,8,12),0,16,-4,3)

        x = fi(pi, 1, 18, 4)
        check(x, 0.3, 1, 18, 4, 167)
        check(x, -0.3, 1, 18, 4, -167)

    def test_iop(self):
        x = fi(1.12345,1,16,7)
        x += 0.5231
        self.assertEqual(x,1.648437500000000)
        self.assertEqual(x.w,17)
        self.assertEqual(x.f,7)

    def test_fixed_M(self):
        q = [0.814723686393179,0.905791937075619,0.126986816293506]
        a = fi(q,1,16,8,FullPrecision=True)
        a3 = a / 0.3333
        self.assertTrue(np.all(a3.int==[627,   696,    99]))
        self.assertEqual(a3.w,16)
        self.assertEqual(a3.f,8)
        
    def test_neg(self):
        x = fi([1,2,3],1,16,8)
        self.assertTrue(np.all(-x==[-1,-2,-3]))
        self.assertEqual(x.s,1)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

        x = fi([1,2,3],0,16,8)
        self.assertTrue(np.all(-x==[0,0,0]))
        self.assertEqual(x.s,0)
        self.assertEqual(x.w,16)
        self.assertEqual(x.f,8)

    def test_invert(self):
        x = fi([1,2,3],1,16,8)
        x_invert_bin = np.array([np.binary_repr(~i,x.w) for i in x.int])
        y = ~x 
        self.assertTrue(np.all(y.bin==x_invert_bin))
        self.assertEqual(y.s,1)
        self.assertEqual(y.w,16)
        self.assertEqual(y.f,8)

        x = fi([1,2,3],0,16,8)
        y = ~x
        self.assertTrue(np.all(y.bin==x_invert_bin))
        self.assertEqual(y.s,0)
        self.assertEqual(y.w,16)
        self.assertEqual(y.f,8)

    def test_pow(self):
        x = fi([0,1+1/77,-3-52/123],1,16,8)
        y = x**3
        self.assertTrue(np.all(y==[  0.        ,   1.03515625, -40.06640625]))
        self.assertEqual(y.s,1)
        self.assertEqual(y.w,16)
        self.assertEqual(y.f,8)
    
    def test_bitwise(self):
        n = np.array([0b1101,0b1001,0b0001,0b1111])/2**8
        x = fi(n,1,16,8)
        self.assertTrue(np.all((x & 0b1100).int==[0b1100,0b1000,0b0000,0b1100]))
        self.assertTrue(np.all((0b0101 | x).int==[0b1101,0b1101,0b0101,0b1111]))
        self.assertTrue(np.all((x ^      x).int==[0b0000,0b0000,0b0000,0b0000]))
        self.assertTrue(np.all((x >>     1).int==[0b0110,0b0100,0b0000,0b0111]))
        self.assertTrue(np.all((x <<     1).int==[0b11010,0b10010,0b00010,0b11110]))
        self.assertTrue(np.all((x <<    14).int==[16384, 16384, 16384, -16384])) # shift overflow

    def test_logical(self):
        x = fi([-2,-1,0,1,2])
        self.assertTrue(np.all((x>1)==[False,False,False,False,True]))
        self.assertTrue(np.all((x>=1)==[False,False,False,True,True]))
        self.assertTrue(np.all((x==1)==[False,False,False,True,False]))
        self.assertTrue(np.all((x!=1)==[True,True,True,False,True]))
        self.assertTrue(np.all((x<=1)==[True,True,True,True,False]))
        self.assertTrue(np.all((x<1)==[True,True,True,False,False]))

    def test_ufunc(self):
        x = np.array([0,pi/2,pi,3*pi/2,2*pi])/100
        y = fi(x,1,16,9)
        z = y[0]
        a = np.cos(x)
        b = np.cos(y)
        c = np.cos(y.double)
        d = fi(c)
        self.assertTrue(np.all(b.int==d.int))        
        e = np.arctan2(b,y)
        f = fi(np.arctan2(b.double,y.double))
        self.assertTrue(np.all(e.int==f.int))        

    def test_fi_quantize(self):
        n = np.array([0b11101,0b11001,0b10001,0b11111])
        x = fi(n,1,5,3,quantize=False)
        self.assertTrue(np.all(x.bin == ['11101','11001','10001','11111']))

        y = fi(n,1,4,2,quantize=False)
        self.assertTrue(np.all(y.bin == ['1101','1001','0001','1111']))

    def test_complex(self):
        x = np.random.randn(10) + 1j*np.random.randn(10)
        a = fi(x,1,16)
        b = fi(x.tolist(),1,16)
        self.assertTrue(np.all(np.abs(x-b.double)))
    
    def test_complex_fft(self):
        x = np.random.randn(10) + 1j*np.random.randn(10)
        a = fi(x,1,32,16)
        fa = np.fft.fft(a)
        b = a.double
        fb = np.fft.fft(b)
        self.assertTrue(np.all(np.abs(fa.double-fb)<1e-5))

if __name__ == '__main__':
    unittest.main()