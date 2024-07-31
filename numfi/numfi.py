import numpy as np
from typing import Literal
type RoundingMethod_Enum = Literal['Nearest', 'Round', 'Convergent','Floor','Zero','Ceiling']
type OverflowAction_Enum = Literal['Error','Wrap','Saturate']

class numfi_tmp(np.ndarray):
    def __new__(cls, array=[], s:int|None|bool=None, w:int|None=None, f:int|None=None, quantize:bool=True, **kwargs) -> 'numfi_tmp':
        # priority: explicit like > array
        like = kwargs.get('like', array)
        # priority: explicit args > like.attr > default(1,32,16,'Nearest',Saturate,False)
        s = round(s) if isinstance(s, (int, float, np.integer, np.floating)) else getattr(like, 's', 1)
        w = round(w) if isinstance(w, (int, float, np.integer, np.floating)) else getattr(like, 'w', 16)
        f = round(f) if isinstance(f, (int, float, np.integer, np.floating)) else getattr(like, 'f', numfi_tmp.get_best_precision(array,s,w))
        RoundingMethod = kwargs.get('RoundingMethod',getattr(like, 'RoundingMethod', 'Nearest'))
        OverflowAction = kwargs.get('OverflowAction',getattr(like, 'OverflowAction', 'Saturate'))
        FullPrecision = bool(kwargs.get('FullPrecision',getattr(like, 'FullPrecision', True)))
        assert w > 0, f"w must be positive integer, but get {w}"

        if np.iscomplexobj(array):
            iarray = cls.__quantize__(np.real(array), s, w, f, RoundingMethod, OverflowAction, quantize) + \
                1j * cls.__quantize__(np.imag(array), s, w, f, RoundingMethod, OverflowAction, quantize)
        else:            
            iarray = cls.__quantize__(array,          s, w, f, RoundingMethod, OverflowAction, quantize)
        obj = iarray.view(cls)
        obj._s, obj._w, obj._f = s, w, f
        obj._RoundingMethod = RoundingMethod
        obj._OverflowAction = OverflowAction
        obj._FullPrecision = FullPrecision
        return obj

    @staticmethod
    def __quantize__(array:np.ndarray, s:int, w:int, f:int,
                     RoundingMethod:RoundingMethod_Enum,
                     OverflowAction:OverflowAction_Enum, quantize:bool=True) -> np.ndarray:        
        if np.shape(array) == ():
            array = np.reshape(array,(1,)) 
        if isinstance(array, numfi_tmp):
            array = array.double
        else:
            np.asarray(array) # scalar to 1d array
        if quantize:
            i = w - f - s
            farray = array.double if isinstance(array, numfi_tmp) else np.asarray(array, dtype=np.float64)
            iarray = farray * 2**f
            if not(isinstance(array, numfi_tmp) and f >= array.f): # no rounding only if new_f >= old_f
                iarray = numfi_tmp.do_rounding(iarray, RoundingMethod)
            if not(isinstance(array, numfi_tmp) and i >= array.i and array.s == s): # no overflow only if new_i >= old_i and new_s == old_s
                iarray = numfi_tmp.do_overflow(iarray, s, w, f, OverflowAction)
        else:
            iarray = np.asarray(array).astype(np.int64)
            iarray &= (1<<w)-1 # truncate to w bits
            if s:
                iarray[iarray >= (1<<(w-1))] -= (1<<w) # two's complement
        return iarray

    def __array_finalize__(self, obj:'numfi_tmp'):
        self._s:int = getattr(obj, 's', 1)
        self._w:int = getattr(obj, 'w', 16)
        self._f:int = getattr(obj, 'f', numfi_tmp.get_best_precision(obj, self.s, self.w))
        self._RoundingMethod:RoundingMethod_Enum = getattr(obj, 'RoundingMethod', 'Nearest')
        self._OverflowAction:OverflowAction_Enum = getattr(obj, 'OverflowAction', 'Saturate')
        self._FullPrecision:bool = getattr(obj, 'FullPrecision', True)

    def __array_ufunc__(self, ufunc:np.ufunc, method, *inputs, out=None, **kwargs) -> 'numfi_tmp':
        # use numpy's ufunc instead of overload bitwise operator like `__or__`, to support situation like `0b101 | x` and `x & y`
        if 'bitwise' in ufunc.__name__ or 'shift' in ufunc.__name__: 
            args = [i.int if isinstance(i, numfi_tmp) else i for i in inputs]
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            results &= (1<<self.w)-1
            results[results >= (1<<(self.w-1))] -= (1<<self.w)
            return type(self)(results, like=self, quantize=False)
        else:
            args = [i.double if isinstance(i, numfi_tmp) else i for i in inputs]
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            return type(self)(results, self.s, self.w) if self.FullPrecision else type(self)(results, like=self)
        
    def __array_function__(self, func, types, args, kwargs):
        args = [i.double if isinstance(i, numfi_tmp) else i for i in args]
        kwargs = {k: v.double if isinstance(v, numfi_tmp) else v for k, v in kwargs.items()}
        results = func(*args, **kwargs)
        return type(self)(results, self.s, self.w) if self.FullPrecision else type(self)(results, like=self)

    def __repr__(self) -> str:
        signed = 's' if self.s else 'u'
        typename = type(self).__name__
        re = typename + self.double.__repr__()[5:]
        return re + f' {signed}{self.w}/{self.f}-{self.RoundingMethod[0]}/{self.OverflowAction[0]}'

    def __getitem__(self, key):
        key = slice(key,key+1,None) if isinstance(key,(int,np.integer)) else key  # in case of key is single integer
        return super().__getitem__(key) # return class with shape (1,) instead of single int/float value

    def __setitem__(self, key, value):
        quantized = type(self)(value, like=self)
        super().__setitem__(key, quantized)            

    def __fixed_arithmetic__(self, func, y):
        raise NotImplementedError("fixed_arithmetic")

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
        if OverflowAction == 'Error':
            if np.any(up | low):
                raise OverflowError(f"Overflow! array[{iarray.min()} ~ {iarray.max()}] > fi({s},{w},{f})[{lower} ~ {upper}]")
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
                s = ''
                b = np.binary_repr(i,width=self.w)
                if self.s:
                    s = f'{b[0]}'
                    b = b[1:]
                if self.i >= 0:
                    return s + b[:self.i].ljust(self.i,'x') + '.' + b[self.i:]
                else:
                    return s + '.' + b.rjust(self.f,'x')
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
    precision:float                     = property(lambda self: 2**-self.f)
    bin:np.ndarray                      = property(lambda self: self.base_repr(2))
    bin_:np.ndarray                     = property(lambda self: self.base_repr(2,frac_point=True))
    oct:np.ndarray                      = property(lambda self: self.base_repr(8))
    dec:np.ndarray                      = property(lambda self: self.base_repr(10))
    hex:np.ndarray                      = property(lambda self: self.base_repr(16))
    data:np.ndarray                     = property(lambda self: self.double)
    Value:str                           = property(lambda self: str(self.double))
    upper:float                         = property(lambda self:  (2**self.i) - self.precision)
    lower:float                         = property(lambda self: -(2**self.i) if self.s else 0)
    ndarray:np.ndarray                  = property(lambda self: self.view(np.ndarray))
    RoundingMethod:RoundingMethod_Enum  = property(lambda self: self._RoundingMethod)
    OverflowAction:OverflowAction_Enum  = property(lambda self: self._OverflowAction)

    @property
    def int(self) -> np.ndarray:
        raise NotImplementedError("int")
    @property
    def double(self) -> np.ndarray:
        raise NotImplementedError("double")

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


class numfi(numfi_tmp):
    """fixed point class that holds float in memory"""
    @staticmethod
    def __quantize__(array:np.ndarray, s:int, w:int, f:int,
                     RoundingMethod:RoundingMethod_Enum,
                     OverflowAction:OverflowAction_Enum, quantize:bool=True) -> np.ndarray:
        array_int = numfi_tmp.__quantize__(array, s, w, f, RoundingMethod, OverflowAction, quantize)
        return array_int * (2**-f)

    @property
    def int(self) -> np.ndarray:
        return (self.ndarray * 2**self.f).astype(np.int64)
    @property
    def double(self) -> np.ndarray:
        return self.ndarray

    def __fixed_arithmetic__(self, func, y):
        s, w, f, i = self.s, self.w, self.f, self.i
        name = func.__name__[-5:-2] # last 3 words of operator name
        in_place = func.__name__[2] == 'i' # __ixxx__ are in place operation, like +=,-=,*=,/=
        if name == 'add' or name == 'sub':
            y_fi = y if isinstance(y, numfi) else type(self)(y,like=self)
            f = max(f, y_fi.f)
            w = max(i, y_fi.i) + f + s + (1 if (s==y_fi.s) else 2)
        else:
            y_fi = y if isinstance(y, numfi_tmp) else type(self)(y, s, w)
            if name == 'mul':
                w = self.w + y_fi.w + (1 if func.__name__ == '__matmul__' else 0)
                f = self.f + y_fi.f
            elif name == 'div':
                if isinstance(y, numfi_tmp):
                    w = max(self.w, y_fi.w)
                    f = self.f - y_fi.f

        float_result = func(self.double, y_fi.double)
        # note that quantization is not needed for full precision mode, new w/f is larger so no precision lost or overflow
        if not (self.FullPrecision or in_place): # if operator is in-place, bits won't grow
            return type(self)(float_result, like=self) # if fixed, quantize full precision result to shorter length
        elif isinstance(y,numfi_tmp) and not y.FullPrecision:
            return type(self)(float_result, like=y)
        else:
            return type(self)(float_result, s, w, f, like=self)

class numqi(numfi_tmp):
    """fixed point class that holds integer in memory"""
    @property
    def int(self) -> np.ndarray:
        return self.ndarray
    @property
    def double(self) -> np.ndarray:
        return self.ndarray * self.precision

    def __fixed_arithmetic__(self, func, y):
        s, w, f, i = self.s, self.w, self.f, self.i
        name = func.__name__[-5:-2] # last 3 words of operator name
        in_place = func.__name__[2] == 'i' # __ixxx__ are in place operation, like +=,-=,*=,/=
        quantize = False
        if name == 'add' or name == 'sub':
            y_fi = y if isinstance(y, numfi_tmp) else type(self)(y,like=self)
            f = max(f, y_fi.f)
            w = max(i, y_fi.i) + f + s + (1 if (s==y_fi.s) else 2)
            a = (self.int * 2**(f-self.f))
            b = (y_fi.int * 2**(f-y_fi.f))
            result = func(a, b)
        elif name == 'mul':
            y_fi = y if isinstance(y, numfi_tmp) else type(self)(y, s, w)
            w = self.w + y_fi.w + (1 if func.__name__ == '__matmul__' else 0)
            f = self.f + y_fi.f
            result = func(self.int, y_fi.int)
        elif name == 'div':
            y_fi = y if isinstance(y, numfi_tmp) else type(self)(y, s, w)
            if isinstance(y, numfi_tmp):
                w = max(self.w, y_fi.w)
                f = self.f - y_fi.f
            result = func(self.double, y_fi.double)
            quantize = True

        # note that quantization is not needed for full precision mode, new w/f is larger so no precision lost or overflow
        if not (self.FullPrecision or in_place): # if operator is in-place, bits won't grow
            return type(self)(result, like=self, quantize=quantize) # if fixed, quantize full precision result to shorter length
        elif isinstance(y,numfi_tmp) and not y.FullPrecision:
            return type(self)(result, like=y, quantize=quantize)
        else:
            return type(self)(result, s, w, f, like=self, quantize=quantize)

