#########################
# numfi - a fixed-point class (mimic from matlab's fixed-point object <fi>) inherited from numpy.ndarray
# author : ZZZZzzzzac zinger.kyon@gmail.com
# version : 0.1
#########################

import numpy as np 

def quantize(array, signed, n_word, n_frac, rounding, overflow):    
    """TODO: docstring""" 
    upper = 2**(n_word-n_frac-bool(signed)) - (2 ** -n_frac)
    lower = -2**(n_word-n_frac-1) if signed else 0 
    flag = (array.f>n_frac) or (array.upper>upper) or (array.lower<lower) if isinstance(array, numfi) else True
    # array.f > n_frac means we will lost precision, so quantize is needed
    # new bound is smaller than array's bound, overflow may happen
    array = np.asarray(array)    
    array = array.reshape(1,) if array.shape == () else array # single value will be convert to (1,) array to allow index

    if flag:
        # quantize method
        array_int = array * (2**n_frac) 
        if rounding == 'round':
            array_int = np.round(array_int).astype(int) # TODO: this is slow, any idea?
        elif rounding == 'floor':
            array_int = np.floor(array_int).astype(int)
        else:
            raise ValueError(f"invaild rounding method: {rounding}")
        # overflow method
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
    # NOTE: array math above can do in place to save memory

class numfi(np.ndarray):
    #region initialization
    def __new__(cls, input_array=[], s=None, w=None, f=None, **kwargs):
        like = kwargs.get('like',None)
        # priority: position args > like.attr > default
        s = s if s is not None else getattr(like, 's', 1)
        w = w if w is not None else getattr(like, 'w', 32)
        f = f if f is not None else getattr(like, 'f', 16)
        rounding = kwargs.get('rounding',getattr(like, 'rounding', 'round')) 
        overflow = kwargs.get('overflow',getattr(like, 'overflow', 'saturate'))
        fixed = bool(kwargs.get('fixed',getattr(like, 'fixed', False)))

        quantized = quantize(input_array, s, w, f, rounding, overflow)
        obj = quantized.view(cls)

        obj.s = s
        obj.w = w
        obj.f = f 
        if obj.i<0:
            raise ValueError("w<f is not supported")
        obj.rounding = rounding
        obj.overflow = overflow
        obj.fixed = fixed
        return obj

    def __array_finalize__(self,obj):
        self.s = getattr(obj, 's', 1)
        self.w = getattr(obj, 'w', 32)
        self.f = getattr(obj, 'f', 16)
        self.rounding = getattr(obj, 'rounding', 'round')
        self.overflow = getattr(obj, 'overflow', 'saturate')
        self.fixed = getattr(obj, 'fixed', False)
        if self.w < self.f:
            raise ValueError("fraction length > word length is not support")
    #endregion initialization
    #region property
    @property
    def ndarray(self):
        return self.view(np.ndarray)
    @property
    def int(self):
        return (self.ndarray * 2**self.f).astype(int)
    @property
    def bin(self): 
        return self.base_repr(2)
    @property 
    def bin_(self):
        return self.base_repr(2,frac_point=True)
    @property
    def hex(self):
        return self.base_repr(16)
    @property
    def i(self):
        return self.w-self.f-bool(self.s)
    @property
    def upper(self):
        return 2**(self.w-self.f-bool(self.s)) - self.precision
    @property
    def lower(self):
        return -2**(self.w-self.f-1) if self.s else 0 
    @property
    def precision(self):
        return 2**(-self.f)
    #endregion property
    #region methods
    def base_repr(self, base=2, frac_point=False): # return ndarray with same shape and dtype = '<Uw' where w=self.w
        if base == 2:
            add_point = lambda s: s[:-self.f] + '.' + s[-self.f:] if frac_point else s
            func = lambda i: add_point(np.binary_repr(i,width=self.w))    
        else:
            func = lambda i: np.base_repr(i,base=base)
        return np.vectorize(func)(self.int)        
    #endregion methods    
    #region overload operators
    def __repr__(self):
        signed = 's' if self.s else 'u'
        return f'numfi-{signed}{self.w}/{self.f}: \n' + self.ndarray.__repr__() 
    def __getitem__(self,key):
        item = super().__getitem__(key) # return numfi with shape (1,) instead of single value
        return item if isinstance(item, numfi) else numfi(item, like=self)
    def __setitem__(self, key, item):
        quantized = quantize(item, self.s, self.w, self.f, self.rounding, self.overflow)
        super().__setitem__(key,quantized)
    #endregion overload operators
    #region overload arithmetic operators     
    def __arithmeticA__(self, func, y):        
        y = y if isinstance(y, numfi) else numfi(y, like=self)
        i = max(self.i, y.i)
        f = max(self.f, y.f)
        s = 1 if self.s | y.s else 0
        result = numfi(func(y), s, i+f+s+1, f, like=self) # full precision - no need to quantize since f/i is extented  
        if self.fixed: 
            return numfi(result,like=self) # if fixed, quantize full precision result to fixed length
        elif y.fixed:
            return numfi(result,like=y)            
        else: 
            return result
    __add__         = lambda self,y: self.__arithmeticA__(super().__add__, y)
    __radd__        = lambda self,y: self.__arithmeticA__(super().__radd__, y)
    __iadd__        = lambda self,y: self.__arithmeticA__(super().__iadd__, y)
    __sub__         = lambda self,y: self.__arithmeticA__(super().__sub__, y)
    __rsub__        = lambda self,y: self.__arithmeticA__(super().__rsub__, y)
    __isub__        = lambda self,y: self.__arithmeticA__(super().__isub__, y)

    def __arithmeticM__(self, func, y):
        y = y if isinstance(y, numfi) else numfi(y, like=self)
        result = numfi(func(y).view(np.ndarray), self.s|y.s, self.w+y.w, self.f+y.f, like=self) # TODO: mul/div need quantize
        if self.fixed:
            return numfi(result, like=self)
        elif y.fixed:
            return numfi(result, like=y)
        else:
            return result
    __mul__         = lambda self,y: self.__arithmeticM__(super().__mul__, y)
    __rmul__        = lambda self,y: self.__arithmeticM__(super().__rmul__, y)
    __imul__        = lambda self,y: self.__arithmeticM__(super().__imul__, y)
    __truediv__     = lambda self,y: self.__arithmeticM__(super().__truediv__, y)
    __rtruediv__    = lambda self,y: self.__arithmeticM__(super().__rtruediv__, y)
    __itruediv__    = lambda self,y: self.__arithmeticM__(super().__itruediv__, y)
    __floordiv__    = lambda self,y: self.__arithmeticM__(super().__floordiv__, y)
    __rfloordiv__   = lambda self,y: self.__arithmeticM__(super().__rfloordiv__, y)
    __ifloordiv__   = lambda self,y: self.__arithmeticM__(super().__ifloordiv__, y)

    __neg__         = lambda self:   numfi(-self.ndarray, like=self)
    __invert__      = lambda self,y: numfi(~self.ndarray, like=self)
    __pow__         = lambda self,y: numfi(self.ndarray ** y, like=self)
    __mod__         = lambda self,y: numfi(self.ndarray %  y, like=self)
    __lshift__      = lambda self,y: numfi(self.ndarray << y, like=self) # TODO: l/rshift, how?
    __rshift__      = lambda self,y: numfi(self.ndarray >> y, like=self)
    __and__         = lambda self,y: numfi(self.ndarray &  y, like=self) 
    __or__          = lambda self,y: numfi(self.ndarray |  y, like=self) 
    __xor__         = lambda self,y: numfi(self.ndarray ^  y, like=self) 

    __eq__          = lambda self,y: self.ndarray == y
    __ne__          = lambda self,y: self.ndarray != y
    __ge__          = lambda self,y: self.ndarray >= y
    __gt__          = lambda self,y: self.ndarray >  y
    __le__          = lambda self,y: self.ndarray <= y
    __lt__          = lambda self,y: self.ndarray <  y
    #endregion