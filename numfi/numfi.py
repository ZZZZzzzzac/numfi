import numpy as np
from typing import Literal
import warnings
RoundingMethod_Enum = Literal['Nearest', 'Round', 'Convergent','Floor','Zero','Ceiling']
OverflowAction_Enum = Literal['Error','Wrap','Saturate']

def lshift(x, s):
    return x << s if s >= 0 else x >> -s

class numfi(np.ndarray):
    def __new__(cls, array=[], s:int|None|bool=None, w:int|None=None, f:int|None=None, **kwargs) -> 'numfi':
        # priority: explicit like > array
        like = kwargs.get('like', array)
        if issubclass(type(array),numfi):
            array = array.double
        # priority: explicit args > like.attr > default(1,32,16,'Nearest',Saturate,False)
        s = round(s) if isinstance(s, (int, float, np.integer, np.floating)) else getattr(like, 's', 1)
        w = round(w) if isinstance(w, (int, float, np.integer, np.floating)) else getattr(like, 'w', 16)
        f = round(f) if isinstance(f, (int, float, np.integer, np.floating)) else getattr(like, 'f', numfi.get_best_precision(array,s,w))
        RoundingMethod = str(kwargs.get('RoundingMethod',getattr(like, 'RoundingMethod', 'Nearest')))
        OverflowAction = str(kwargs.get('OverflowAction',getattr(like, 'OverflowAction', 'Saturate')))
        FullPrecision = bool(kwargs.get('FullPrecision', getattr(like, 'FullPrecision',  True)))
        quantize =      bool(kwargs.get('quantize',      getattr(like, 'quantize',       True)))
        iscpx =         bool(kwargs.get('iscpx',         getattr(like, 'iscpx',          False)))
        assert w > 0, f"w must be positive integer, but get {w}"

        array = np.asarray(array)
        if np.iscomplexobj(array):
            iscpx = True
        if iscpx:
            iarray = cls.quantize(array.real, s, w, f, RoundingMethod, OverflowAction, quantize) + \
                1j * cls.quantize(array.imag, s, w, f, RoundingMethod, OverflowAction, quantize)
        else:
            iarray = cls.quantize(array,      s, w, f, RoundingMethod, OverflowAction, quantize)
        obj = iarray.view(cls)
        obj._s, obj._w, obj._f = s, w, f
        obj._RoundingMethod = RoundingMethod
        obj._OverflowAction = OverflowAction
        obj._FullPrecision = FullPrecision
        obj._iscpx = iscpx
        return obj

    @staticmethod
    def quantize(array:np.ndarray, s:int, w:int, f:int,
                RoundingMethod:RoundingMethod_Enum,
                OverflowAction:OverflowAction_Enum, quantize:bool=True) -> np.ndarray:        
        if np.shape(array) == ():
            array = np.reshape(array,(1,)) # scalar to 1d array
        if isinstance(array, numfi):
            array = array.double
        if quantize:
            i = w - f - s
            farray = array.double if isinstance(array, numfi) else np.asarray(array, dtype=np.float64)
            iarray = farray * 2**f
            if not(isinstance(array, numfi) and f >= array.f): # no rounding only if new_f >= old_f
                iarray = numfi.do_rounding(iarray, RoundingMethod)
            if not(isinstance(array, numfi) and i >= array.i and array.s == s): # no overflow only if new_i >= old_i and new_s == old_s
                iarray = numfi.do_overflow(iarray, s, w, f, OverflowAction)
        else:
            iarray = np.asarray(array).astype(np.int64)
            iarray &= (1<<w)-1 # truncate to w bits
            if s:
                iarray[iarray >= (1<<(w-1))] -= (1<<w) # two's complement
        return iarray

    def __array_finalize__(self, obj:'numfi'):
        self._s:int = getattr(obj, 's', 1)
        self._w:int = getattr(obj, 'w', 16)
        self._f:int = getattr(obj, 'f', numfi.get_best_precision(obj, self.s, self.w))
        self._RoundingMethod:RoundingMethod_Enum = getattr(obj, 'RoundingMethod', 'Nearest')
        self._OverflowAction:OverflowAction_Enum = getattr(obj, 'OverflowAction', 'Saturate')
        self._FullPrecision:bool = getattr(obj, 'FullPrecision', True)
        self._iscpx:bool = getattr(obj, 'iscpx', False)

    def __array_ufunc__(self, ufunc:np.ufunc, method, *inputs, out=None, **kwargs) -> 'numfi':
        # use numpy's ufunc instead of overload bitwise operator like `__or__`, to support situation like `0b101 | x` and `x & y`
        if 'bitwise' in ufunc.__name__ or 'shift' in ufunc.__name__: 
            args = [i.int if isinstance(i, numfi) else i for i in inputs]
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            results &= (1<<self.w)-1
            results[results >= (1<<(self.w-1))] -= (1<<self.w)
            return type(self)(results, like=self, quantize=False)
        else:
            args = [i.double if isinstance(i, numfi) else i for i in inputs]
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            return type(self)(results, self.s, self.w) if self.FullPrecision else type(self)(results, like=self)
        
    def __array_function__(self, func, types, args, kwargs):
        args = [i.double if isinstance(i, numfi) else i for i in args]
        kwargs = {k: v.double if isinstance(v, numfi) else v for k, v in kwargs.items()}
        results = func(*args, **kwargs)
        return type(self)(results, self.s, self.w) if self.FullPrecision else type(self)(results, like=self)

    def __repr__(self) -> str:
        signed = 's' if self.s else 'u'
        typename = type(self).__name__
        re = typename + self.double.__repr__()[5:]
        return re + f' {signed}{self.w}/{self.f}-{self.RoundingMethod[0]}/{self.OverflowAction[0]}'
    
    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return v if issubclass(type(v),numfi) else type(self)(v, quantize=False, like=self)
    
    def __setitem__(self, key, value):
        quantized = type(self)(value, like=self)
        if isinstance(key, (int,np.integer)): # suppress DeprecationWarning
            key = slice(key,key+1,None)
        super().__setitem__(key, quantized)          

    def __fixed_arithmetic__(self, func, y):
        s, w, f = self.s, self.w, self.f
        name = func.__name__[-5:-2]  # last 3 chars of operator name

        if name in ('add', 'sub'):
            y = y if isinstance(y, numfi) else type(self)(y, like=self)
            s = max(s, y.s)
            if self.FullPrecision and y.FullPrecision:
                w = max(w - f, y.w - y.f) + (1 + (self.s ^ y.s)) + max(f, y.f)
                f = max(f, y.f)
            else:
                w, f = max(w, y.w), max(f, y.f)
            result = func(self.int << (f - self.f), y.int << (f - y.f))
            return type(self)(result, s, w, f, like=self, quantize=False)
            
        elif name == 'mul':
            y = y if isinstance(y, numfi) else type(self)(y,s,w)
            s = max(s, y.s)
            if self.FullPrecision and y.FullPrecision:
                w = self.w + y.w + (func.__name__ == '__matmul__')
                f = self.f + y.f
                return type(self)(func(self.int, y.int), s, w, f, like=self, quantize=False)
            else:
                w, f = max(w, y.w), max(f, y.f)
            return type(self)(func(self.int, y.int) >> y.f, s, w, f, like=self, quantize=False)
        
        elif name == 'div':
            if isinstance(y, numfi):
                s = max(s, y.s)
                if self.FullPrecision and y.FullPrecision:
                    w = max(self.w, y.w)
                    f = self.f - y.f
                    return type(self)(np.round(func(self.int, y.int)).astype(np.int64), 
                                    s, w, f, like=self, quantize=False)
                return type(self)(func(self.double, y.double), s, w, f, like=self)
            return type(self)(func(self.double, y), like=self)

    @staticmethod
    def do_rounding(iarray:np.ndarray, RoundingMethod:RoundingMethod_Enum) -> np.ndarray:
        if RoundingMethod in ('Nearest', 'Round', 'Convergent'):  # round towards nearest integer, this method is faster than np.round()
            iarray[iarray>0] += 0.5
            iarray[iarray<0] -= 0.5
        elif RoundingMethod == 'Floor': # round towards -inf
            iarray = np.floor(iarray)
        elif RoundingMethod == 'Zero': # round towards zero, fastest rounding method
            pass
        elif RoundingMethod == 'Ceiling':  # round towards +inf
            iarray = np.ceil(iarray)
        else:
            raise ValueError(f"invaild RoundingMethod: {RoundingMethod}")
        return iarray.astype(np.int64)

    @staticmethod
    def do_overflow(iarray:np.ndarray, s:int, w:int, f:int, OverflowAction:OverflowAction_Enum) -> np.ndarray:
        iarray = iarray.astype(np.int64)
        upper =  (1<<(w-s)) - 1
        lower = -(1<<(w-s)) if s else 0
        up =  iarray > upper
        low = iarray < lower
        if np.any(up | low) and OverflowAction != 'Ignore':
            msg = f"Overflow! array[{iarray.min()} ~ {iarray.max()}] > fi({s},{w},{f})[{lower} ~ {upper}]"
            if OverflowAction == 'Error':
                raise OverflowError(msg)
            elif OverflowAction == 'warning':
                warnings.warn(msg, RuntimeWarning)
            elif OverflowAction == 'Wrap':
                mask = (1<<w)
                iarray &= (mask-1)
                if s:
                    iarray[iarray>=(1<<(w-1))] |= (-mask)
            elif OverflowAction == 'Saturate':
                iarray[up] = upper
                iarray[low] = lower
            else:
                raise ValueError(f"invaild OverflowAction: {OverflowAction}")
        return iarray

    @staticmethod
    def get_best_precision(x, s:int, w:int) -> int:
        if np.iscomplexobj(x):
            x = np.asarray(x, dtype=np.complex128)
            maximum = max(np.max(x.real), np.max(x.imag)) if np.size(x) else 0
            minimum = min(np.min(x.real), np.min(x.imag)) if np.size(x) else 0
        else:
            x = np.asarray(x, dtype=np.float64)
            maximum = np.max(x) if np.size(x) else 0
            minimum = np.min(x) if np.size(x) else 0
        if not (maximum==minimum==0):
            if maximum > -minimum:
                return int(w - np.floor(np.log2(maximum)) - 1 - s)
            else:
                return int(w - np.ceil(np.log2(-minimum)) - s)
        else:
            return 15

    def base_repr(self, base:int=2, frac_point:bool=False) -> np.ndarray: # return ndarray with same shape and dtype = '<Uw' where w=self.w
        if base == 2:
            def pretty_bin(i):
                b = np.binary_repr(i,width=self.w)
                l = self.w - self.f
                if l < 0:
                    return '.'+b.rjust(self.f,'x')
                return b[:l].ljust(l,'x') + '.' + b[l:]
            func = pretty_bin if frac_point else lambda i: np.binary_repr(i,self.w)
        else:
            def func(i):
                s = np.base_repr(i if i>=0 else i+(1<<self.w),base=base)
                b = int(np.ceil(self.w/4))
                return "0"*(b-len(s)) + s
        if self.size > 0:
            return np.vectorize(func)(self.int)
        else:
            return np.array([],dtype=f"<U{self.w}")

    s:int                               = property(lambda self: self._s)
    w:int                               = property(lambda self: self._w)
    f:int                               = property(lambda self: self._f)
    i:int                               = property(lambda self: self.w - self.f - self.s)
    FullPrecision:bool                  = property(lambda self: self._FullPrecision)
    RoundingMethod:RoundingMethod_Enum  = property(lambda self: self._RoundingMethod)
    OverflowAction:OverflowAction_Enum  = property(lambda self: self._OverflowAction)
    iscpx                               = property(lambda self: self._iscpx)
    ndarray:np.ndarray                  = property(lambda self: self.view(np.ndarray))
    data:np.ndarray                     = property(lambda self: self.double)
    bin:np.ndarray                      = property(lambda self: self.base_repr(2))
    bin_:np.ndarray                     = property(lambda self: self.base_repr(2,frac_point=True))
    oct:np.ndarray                      = property(lambda self: self.base_repr(8))
    dec:np.ndarray                      = property(lambda self: self.base_repr(10))
    hex:np.ndarray                      = property(lambda self: self.base_repr(16))
    upper:float                         = property(lambda self:  (2**self.i) - self.precision)
    lower:float                         = property(lambda self: -(2**self.i) if self.s else 0)
    precision:float                     = property(lambda self: 2**-self.f)
    Value:str                           = property(lambda self: str(self.double))

    @property
    def int(self) -> np.ndarray:
        return self.ndarray
    @property
    def double(self) -> np.ndarray:
        if self.iscpx:
            return (self.ndarray * self.precision).view(np.complex128)
        return self.ndarray * self.precision

    __round__       = lambda self,y=0: np.round(self,y)

    __add__         = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__add__,       y)
    __radd__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__radd__,      y)
    __iadd__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__iadd__,      y) # force in place use same memory is not worthy
    __sub__         = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__sub__,       y)
    __rsub__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__rsub__,      y)
    __isub__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__isub__,      y)
    __mul__         = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__mul__,       y)
    __rmul__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__rmul__,      y)
    __imul__        = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__imul__,      y)
    __matmul__      = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__matmul__,    y)
    __truediv__     = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__truediv__,   y)
    __rtruediv__    = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__rtruediv__,  y)
    __itruediv__    = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__itruediv__,  y)
    __floordiv__    = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__floordiv__,  y)
    __rfloordiv__   = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__rfloordiv__, y)
    __ifloordiv__   = lambda self,y: self.__fixed_arithmetic__(np.ndarray.__ifloordiv__, y)

    __neg__         = lambda self:   type(self)(-self.double,      like=self)
    __pow__         = lambda self,y: type(self)( self.double ** y, like=self)
    __mod__         = lambda self,y: type(self)( self.double %  y, like=self)
    __invert__      = lambda self:   type(self)(~self.int, like=self, quantize=False)
    # two numfi comparison: `x>y == x.__gt__(self,y) == x.double>y == y<x.double == y.__lt__(self,x.double) == y.double < x.double` => two float comparison
    __eq__          = lambda self,y: self.double == y
    __ne__          = lambda self,y: self.double != y
    __ge__          = lambda self,y: self.double >= y
    __gt__          = lambda self,y: self.double >  y
    __le__          = lambda self,y: self.double <= y
    __lt__          = lambda self,y: self.double <  y
