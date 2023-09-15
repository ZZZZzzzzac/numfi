#########################
# numfi - a numpy.ndarray subclass that does fixed-point arithmetic.
# author : zinger.kyon@gmail.com
#########################

#TODO: complex support?
#TODO: sin(numfi) return error larger than sin(fxpmath), further study on __array_ufunc__ is needed
import numpy as np 
import warnings


def do_rounding(array_int, rounding):
    if rounding in ('Nearest', 'Round', 'Convergent'):  # round towards nearest integer, this method is faster than np.round()
        array_int[array_int>0] += 0.5
        array_int[array_int<0] -= 0.5
    elif rounding == 'Floor': # round towards -inf
        array_int = np.floor(array_int)
    elif rounding == 'Zero': # round towards zero, fastest rounding method
        pass
    elif rounding == 'Ceiling':  # round towards +inf
        array_int = np.ceil(array_int)
    else:
        raise ValueError(f"invaild rounding method: {rounding}")
    return array_int.astype(np.int64)

def do_overflow(array_int, s, w, is_saturate, is_raise=False):
    array_int = array_int.astype(np.int64)
    # raise exception if overflow
    upper =  (1<<(w-s)) - 1 
    lower = -(1<<(w-s)) if s else 0
    up = array_int>upper
    low = array_int<lower
    if is_raise:
        if np.any(up) or np.any(low):
            raise OverflowError(f"overflow detected, w[{w}], s[{s}]")
    
    if is_saturate:
        array_int[array_int>upper] = upper
        array_int[array_int<lower] = lower
    else: # worst 3n, best 2n    
        mask = (1<<w)
        array_int &= (mask-1)
        if s:
            array_int[array_int>=(1<<(w-1))] |= (-mask)    
    return array_int


class numfi_tmp(np.ndarray):
    def __new__(cls, array=[], s=None, w=None, f=None, **kwargs):
        # priority: explicit like > array
        like = kwargs.get('like', array)
        # priority: explicit args > like.attr > default(1,32,16)
        s = s if isinstance(s, (int, np.integer)) else getattr(like, 's', 1)
        w = w if isinstance(w, (int, np.integer)) else getattr(like, 'w', 32)
        f = f if isinstance(f, (int, np.integer)) else getattr(like, 'f', 16)
        rounding = kwargs.get('rounding',getattr(like, 'rounding', 'round')) 
        overflow = kwargs.get('overflow',getattr(like, 'overflow', 'saturate'))
        fixed = bool(kwargs.get('fixed',getattr(like, 'fixed', False)))
        assert w > 0, f"numfi_tmp.w[{w}] must be positive integer"

        obj = cls.__quantize__(array, s, w, f, rounding, overflow).view(cls)
        obj._s, obj._w, obj._f = int(bool(s)), w, f
        obj._rounding = rounding
        obj._overflow = overflow
        obj._fixed = fixed
        obj._precision = 2 ** (-obj._f)
        return obj
    
    @staticmethod
    def __quantize__(array, s, w, f, rounding, overflow):
        s = int(bool(s)) 
        farray = np.asarray(array,dtype=np.float64)
        farray = farray.reshape(1,) if farray.shape == () else farray
        array_int = farray * (2**f)
        if not(isinstance(array, numfi_tmp) and array.f <= f):
            array_int = do_rounding(array_int, rounding)
        if not(isinstance(array, numfi_tmp) and array.i <= (w - f - s)):
            array_int = do_overflow(array_int, s, w, overflow)
        return array_int

    def __array_finalize__(self, obj):
        self._s = getattr(obj, 's', 1)
        self._w = getattr(obj, 'w', 32)
        self._f = getattr(obj, 'f', 16)
        self._rounding = getattr(obj, 'rounding', 'round')
        self._overflow = getattr(obj, 'overflow', 'saturate')
        self._fixed = getattr(obj, 'fixed', False)
        self._precision = 2 ** (-self._f)

    s           = property(lambda self: self._s)
    w           = property(lambda self: self._w)
    f           = property(lambda self: self._f)
    i           = property(lambda self: self._w - self._f - self._s)
    rounding    = property(lambda self: self._rounding)
    overflow    = property(lambda self: self._overflow)
    fixed       = property(lambda self: self._fixed)
    precision   = property(lambda self: self._precision)

    bin         = property(lambda self: self.base_repr(2))
    bin_        = property(lambda self: self.base_repr(2,frac_point=True))
    hex         = property(lambda self: self.base_repr(16))
    upper       = property(lambda self:  (2**self.i) - self.precision)
    lower       = property(lambda self: -(2**self.i) if self.s else 0)

    @property
    def ndarray(self):
        view = self.view(np.ndarray)
        view.flags.writeable = False # ndarray is read only to avoid undesired modify
        return view    
    @property
    def int(self):
        raise NotImplementedError("int")
    @property
    def float(self):
        raise NotImplementedError("float")

    def base_repr(self, base=2, frac_point=False): # return ndarray with same shape and dtype = '<Uw' where w=self.w
        if base == 2:
            add_point = lambda s: s[:-self.f] + '.' + s[-self.f:] if frac_point else s
            func = lambda i: add_point(np.binary_repr(i,width=self.w))    
        else:
            func = lambda i: np.base_repr(i,base=base)
        if self.size>0:
            return np.vectorize(func)(self.int)        
        else:
            return np.array([],dtype=f"<U{self.w}")

    def __repr__(self):
        signed = 's' if self.s else 'u'
        typename = type(self).__name__
        re = typename + self.float.__repr__()[5:]
        return re + f' {signed}{self.w}/{self.f}-{self.rounding[0]}/{self.overflow[0]}'

    def __getitem__(self,key):
        value = super().__getitem__(key) # return class with shape (1,) instead of single int/float value
        return value if isinstance(value, numfi_tmp) else type(self)(value, like=self)

    def __setitem__(self, key, value):
        quantized = type(self)(value, like=self)
        super().__setitem__(key, quantized)

    def __fixed_arithmetic__(self, func, y):
        raise NotImplementedError("__fixed_arithmetic__")

    __round__       = lambda self,y=0: np.round(self,y)

    __add__         = lambda self,y: self.__fixed_arithmetic__(super().__add__,       y)
    __radd__        = lambda self,y: self.__fixed_arithmetic__(super().__radd__,      y)    
    __iadd__        = lambda self,y: self.__fixed_arithmetic__(super().__iadd__,      y) # force in place use same memory is not worthy
    __sub__         = lambda self,y: self.__fixed_arithmetic__(super().__sub__,       y)
    __rsub__        = lambda self,y: self.__fixed_arithmetic__(super().__rsub__,      y) 
    __isub__        = lambda self,y: self.__fixed_arithmetic__(super().__isub__,      y)   
    __mul__         = lambda self,y: self.__fixed_arithmetic__(super().__mul__,       y)
    __rmul__        = lambda self,y: self.__fixed_arithmetic__(super().__rmul__,      y)    
    __imul__        = lambda self,y: self.__fixed_arithmetic__(super().__imul__,      y)
    __matmul__      = lambda self,y: self.__fixed_arithmetic__(super().__matmul__,    y)
    __truediv__     = lambda self,y: self.__fixed_arithmetic__(super().__truediv__,   y)
    __rtruediv__    = lambda self,y: self.__fixed_arithmetic__(super().__rtruediv__,  y)  
    __itruediv__    = lambda self,y: self.__fixed_arithmetic__(super().__itruediv__,  y)  
    __floordiv__    = lambda self,y: self.__fixed_arithmetic__(super().__floordiv__,  y)
    __rfloordiv__   = lambda self,y: self.__fixed_arithmetic__(super().__rfloordiv__, y)
    __ifloordiv__   = lambda self,y: self.__fixed_arithmetic__(super().__ifloordiv__, y)

    __neg__         = lambda self:   type(self)(-self.float,      like=self)
    __pow__         = lambda self,y: type(self)( self.float ** y, like=self)
    __mod__         = lambda self,y: type(self)( self.float %  y, like=self)
    # bit wise operation use self.int and convert back
    __invert__      = lambda self:   type(self)((~self.int)      * self.precision, like=self) # bitwise invert in two's complement 
    __and__         = lambda self,y: type(self)(( self.int &  y) * self.precision, like=self) 
    __or__          = lambda self,y: type(self)(( self.int |  y) * self.precision, like=self) 
    __xor__         = lambda self,y: type(self)(( self.int ^  y) * self.precision, like=self) 
    __lshift__      = lambda self,y: type(self)(( self.int << y) * self.precision, like=self)
    __rshift__      = lambda self,y: type(self)(( self.int >> y) * self.precision, like=self)

    __eq__          = lambda self,y: self.float == y
    __ne__          = lambda self,y: self.float != y
    __ge__          = lambda self,y: self.float >= y
    __gt__          = lambda self,y: self.float >  y
    __le__          = lambda self,y: self.float <= y
    __lt__          = lambda self,y: self.float <  y





#TODO: large integer may overflow due to dtype=float64 lost precision, np.int64(np.float64(np.int64(9223372036854775807))) = -9223372036854775808


class numfi(numfi_tmp): 
    """fixed point class that holds float in memory"""
    @staticmethod
    def __quantize__(array, s, w, f, rounding, overflow):
        array_int = numfi_tmp.__quantize__(array, s, w, f, rounding, overflow)
        return array_int * (2**-f)
    @property
    def int(self):
        return (self.ndarray * 2**self.f).astype(np.int64)
    @property
    def float(self):
        return self.ndarray
  
    def __fixed_arithmetic__(self, func, y):
        y_fi = y if isinstance(y, numfi) else numfi(y, like=self)
        s, w, f = int(self.s or y_fi.s), self.w, self.f
        name = func.__name__[-5:-2]
        in_place = func.__name__[2] == 'i' # __ixxx__ are in place operation, like +=,-=,*=,/=
        if name == 'add' or name == 'sub': 
            f = max(self.f, y_fi.f)
            w = max(self.i, y_fi.i) + f + s + 1
        elif name == 'mul':
            w = self.w + y_fi.w + (1 if func.__name__ == '__matmul__' else 0)
            f = self.f + y_fi.f
        elif name == 'div':
            if not isinstance(y, numfi):
                y_fi.view(np.ndarray)[:] = y
            else:
                w = max(self.w, y.w)
                f = self.f - y.f
        result = numfi(func(y_fi).float, s, w, f, like=self)
        # note that quantization is not needed for full precision mode here, new w/f is larger so no precision lost or overflow
        if self.fixed or in_place: # if operator is in-place, bits won't grow
            return numfi(result,like=self) # if fixed, quantize full precision result to shorter length
        elif y_fi.fixed:
            return numfi(result,like=y_fi)            
        else: 
            return result







class numqi(numfi_tmp):
    """fixed point class that holds integer in memory """
    @staticmethod
    def __quantize__(array, s, w, f, rounding, overflow):
        if isinstance(array, numqi):
            array_int = array
        else:
            array_int = numfi_tmp.__quantize__(array, s, w, f, rounding, overflow)
        return array_int
    @property
    def int(self):
        return self.ndarray
    @property
    def float(self):
        return self.ndarray.astype(np.float64) * self.precision

    def __fixed_arithmetic__(self, func, y):
        if isinstance(y, numqi):
            if self.s != y.s or self.w != y.w or self.f != y.f:
                raise ArithmeticError("cannot do arithmetic on two different numqi format")
            y_qi = y
        else:
            y_qi = numqi(y, like=self)
        name = func.__name__[-5:-2]
        if name == 'add' or name == 'sub':
            result = func(y_qi)
        elif name == 'mul':
            result = func(y_qi) >> self.f
        elif name == 'div':
            result = getattr(np.ndarray,func.__name__)(self.int << self.f, y_qi)
        qi = self.copy()
        qi.view(np.ndarray)[:] = do_overflow(result.int, self.s, self.w, self.overflow)
        return qi