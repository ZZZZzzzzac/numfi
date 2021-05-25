#########################
# numfi - a numpy.ndarray subclass that does fixed-point arithmetic.
# author : zinger.kyon@gmail.com
#########################

#TODO: complex support?
#TODO: sin(numfi) return error larger than sin(fxpmath), further study on __array_ufunc__ is needed
import numpy as np 
import warnings

def quantize(array, signed, n_word, n_frac, rounding, overflow):   
    bound = 2**(n_word-n_frac-(1 if signed else 0))
    upper = bound - 2**-n_frac
    lower = -2**(n_word-n_frac-1) if signed else 0 

    if upper == bound: # upper==bound means nfrac is too large that precision less than floating point resolution
        warnings.warn(f"n_frac={n_frac} is too large, overflow/underflow may happen during quantization")
    if 2**(n_word-(1 if signed else 0))-1 > np.iinfo(np.int64).max:
        raise OverflowError(f"cannot quantize array, upper/lower overflow np.int64 bound")    

    flag = (array.f>n_frac) or (array.upper>upper) or (array.lower<lower) if isinstance(array, numfi) else True
    # array.f < n_frac means we will not lost precision, so quantization is not need
    array = np.asarray(array,dtype=np.float64) # convert list/numfi/other data type to numpy.ndarray
    #TODO: large integer may overflow due to dtype=float64 lost precision, np.int64(np.float64(np.int64(9223372036854775807))) = -9223372036854775808
    array = array.reshape(1,) if array.shape == () else array # single value will be convert to (1,) array to allow indexing
    
    if flag:
        array_int = array * (2**n_frac) 
        if rounding == 'round':  # round towards nearest integer, this method is faster than np.round()
            array_int[array_int>0]+=0.5
            array_int[array_int<0]-=0.5
            array_int = array_int.astype(np.int64)
        elif rounding == 'floor': # round towards -inf
            array_int = np.floor(array_int).astype(np.int64)
        elif rounding == 'zero': # round towards zero, fastest rounding method
            array_int = array_int.astype(np.int64)
        elif rounding == 'ceil':  # round towards +inf
            array_int = np.ceil(array_int).astype(np.int64)
        else:
            raise ValueError(f"invaild rounding method: {rounding}")

        if overflow == 'wrap': # worst 3n, best 2n
            m = (1<<n_word)
            array_int &= (m-1)
            if signed:
                array_int[array_int>=(1<<(n_word-1))] |= (-m)
            array = array_int * (2 ** -n_frac)
        elif overflow == 'saturate': # worst 4n, best 2n 
            array = array_int * (2 ** -n_frac) 
            array[array>upper] = upper
            array[array<lower] = lower
        else:
            raise ValueError(f"invalid overflow method: {overflow}")

    return array

class numfi(np.ndarray):
    #region initialization
    def __new__(cls, input_array=[], s=None, w=None, f=None, **kwargs):
        like = input_array if isinstance(input_array, numfi) and kwargs.get('like',None) is None else kwargs.get('like',None)
        # priority: position args > like.attr > default
        s = s if isinstance(s, (int, np.integer)) else getattr(like, 's', 1)
        w = w if isinstance(w, (int, np.integer)) else getattr(like, 'w', 32)
        f = f if isinstance(f, (int, np.integer)) else getattr(like, 'f', 16)
        rounding = kwargs.get('rounding',getattr(like, 'rounding', 'round')) 
        overflow = kwargs.get('overflow',getattr(like, 'overflow', 'saturate'))
        fixed = bool(kwargs.get('fixed',getattr(like, 'fixed', False)))

        quantized = quantize(input_array, s, w, f, rounding, overflow)
        obj = quantized.view(cls) # this will call __array_finalize__
        obj._s, obj._w, obj._f = int(bool(s)), w, f
        obj._rounding = rounding
        obj._overflow = overflow
        obj._fixed = fixed
        if not isinstance(obj.w, (int,np.integer)) or obj.w<=0:
            raise ValueError(f"numfi.w[{obj.w}] must be positive integer")
        if not isinstance(obj.f, (int,np.integer)) or obj.f<0:
            raise ValueError(f"numfi.f[{obj.f}] must non negative integer")
        if obj.i<0:
            raise ValueError(f"numfi.i[{obj.i}] = numfi.w - numfi.f - numfi.s must greater than 0")
        return obj

    def __array_finalize__(self,obj):
        self._s = getattr(obj, 's', 1)
        self._w = getattr(obj, 'w', 32)
        self._f = getattr(obj, 'f', 16)
        self._rounding = getattr(obj, 'rounding', 'round')
        self._overflow = getattr(obj, 'overflow', 'saturate')
        self._fixed = getattr(obj, 'fixed', False)
        self._inplace = False
    #endregion initialization
    #region read only property 
    s           = property(lambda self: self._s)
    w           = property(lambda self: self._w)
    f           = property(lambda self: self._f)
    rounding    = property(lambda self: self._rounding)
    overflow    = property(lambda self: self._overflow)
    fixed       = property(lambda self: self._fixed)
    int         = property(lambda self: (self.ndarray * 2**self.f).astype(np.int64))
    bin         = property(lambda self: self.base_repr(2))
    bin_        = property(lambda self: self.base_repr(2,frac_point=True))
    hex         = property(lambda self: self.base_repr(16))
    i           = property(lambda self: self.w-self.f-self.s)
    upper       = property(lambda self: 2**self.i - self.precision)
    lower       = property(lambda self: -2**(self.w-self.f-1) if self.s else 0)
    precision   = property(lambda self: 2**(-self.f))
    @property
    def ndarray(self):
        view = self.view(np.ndarray)
        view.flags.writeable = False # ndarray is read only to avoid undesired modify
        return view    
    #endregion property
    #region methods
    def base_repr(self, base=2, frac_point=False): # return ndarray with same shape and dtype = '<Uw' where w=self.w
        if base == 2:
            add_point = lambda s: s[:-self.f] + '.' + s[-self.f:] if frac_point else s
            func = lambda i: add_point(np.binary_repr(i,width=self.w))    
        else:
            func = lambda i: np.base_repr(i,base=base)
        if self.size>0:
            return np.vectorize(func)(self.int)        
        else:
            raise ValueError("array is empty")
    def requantize(self, *args, **kwargs):
        kwargs['like'] = self if kwargs.get('like') is None else kwargs.get('like')
        return numfi(self,*args, **kwargs)
    #endregion methods    
    #region overload method
    def __repr__(self):
        signed = 's' if self.s else 'u'
        return super().__repr__() + f' {signed}{self.w}/{self.f}-{self.rounding[0]}/{self.overflow[0]}'
    def __getitem__(self,key):
        item = super().__getitem__(key) # return numfi with shape (1,) instead of single value
        return item if isinstance(item, numfi) else numfi(item, like=self)
    def __setitem__(self, key, item):
        quantized = quantize(item, self.s, self.w, self.f, self.rounding, self.overflow)
        super().__setitem__(key,quantized)
    #endregion overload method
    #region overload arithmetic operators     
    def __fixed_arithmetic__(self, func, y):
        y_fi = y if isinstance(y, numfi) else numfi(y, like=self)
        s = self.s | y_fi.s
        name = func.__name__[-5:-2]
        in_place = func.__name__[2] == 'i' # __ixxx__ are in place operation, like +=,-=,*=,/=
        if name == 'add' or name == 'sub': 
            i = max(self.i, y_fi.i)
            f = max(self.f, y_fi.f)            
            result = numfi(func(y_fi), s, i+f+s+1, f, like=self)
        elif name == 'mul':
            if func.__name__ == '__matmul__':
                result = numfi(func(y_fi), s, self.w+y_fi.w+1, self.f+y_fi.f, like=self) # equivalent to mul + add, then word length should be x.w+y.w+1
            else:
                result = numfi(func(y_fi), s, self.w+y_fi.w, self.f+y_fi.f, like=self) 
        elif name == 'div':
            if self.fixed or in_place or not isinstance(y, numfi):
                return numfi(func(y).ndarray, like=self)
            else:        
                if y.fixed:
                    return numfi(func(y).ndarray, like=y)
                else:
                    return numfi(func(y).ndarray, s, max(self.w,y.w), self.f-y.f, like=self) 
        # note that quantization is not needed for full precision mode here, new w/f is larger so no precision lost or overflow
        if self.fixed or in_place: # if operator is in-place, bits won't grow
            return numfi(result,like=self) # if fixed, quantize full precision result to fixed length
        elif y_fi.fixed:
            return numfi(result,like=y_fi)            
        else: 
            return result

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

    __neg__         = lambda self:   numfi(-self.ndarray, like=self)
    __pow__         = lambda self,y: numfi(self.ndarray ** y, like=self)
    __mod__         = lambda self,y: numfi(self.ndarray %  y, like=self)
    # bit wise operation use self.int and convert back
    __invert__      = lambda self:   numfi((~self.int) * self.precision, like=self) # bitwise invert in two's complement 
    __and__         = lambda self,y: numfi((self.int &  y) * self.precision, like=self) 
    __or__          = lambda self,y: numfi((self.int |  y) * self.precision, like=self) 
    __xor__         = lambda self,y: numfi((self.int ^  y) * self.precision, like=self) 
    __lshift__      = lambda self,y: numfi((self.int << y) * self.precision, like=self)
    __rshift__      = lambda self,y: numfi((self.int >> y) * self.precision, like=self)

    __eq__          = lambda self,y: self.ndarray == y
    __ne__          = lambda self,y: self.ndarray != y
    __ge__          = lambda self,y: self.ndarray >= y
    __gt__          = lambda self,y: self.ndarray >  y
    __le__          = lambda self,y: self.ndarray <= y
    __lt__          = lambda self,y: self.ndarray <  y
    #endregion overload arithmetic operators     