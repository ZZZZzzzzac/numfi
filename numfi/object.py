#########################
# numfi - a fixed-point class (mimic from matlab's fixed-point object <fi>) inherited from numpy.ndarray
# author : ZZZZzzzzac zinger.kyon@gmail.com
# version : 0.1
#########################

import numpy as np 
import warnings
from enum import Enum
def quantize(array, signed, n_word, n_frac, rounding, overflow):    
    """docstring""" 
    if n_frac > 32:
        warnings.warn(f"n_frac={n_frac} is very large and may overflow during quantize")
    upper = 2**(n_word-n_frac-bool(signed)) - (2 ** -n_frac)
    lower = -2**(n_word-n_frac-1) if signed else 0 
    flag = (array.f>n_frac) or (array.upper>upper) or (array.lower<lower) if isinstance(array, numfi) else True
    # array.f < n_frac means we will not lost precision, so quantization is not need
    array = np.asarray(array) # convert list/other data type to numpy.ndarray
    if np.issubdtype(array.dtype,np.integer):
        array = array.astype(np.float64)
    array = array.reshape(1,) if array.shape == () else array # single value will be convert to (1,) array to allow index

    if flag:
        # quantize method
        array_int = array * (2**n_frac) 
        if rounding == 'round':  # TODO: other rounding method
            array_int = np.round(array_int).astype(np.int64) # TODO: np.round() is slow, any better idea?
        elif rounding == 'floor':  
            array_int = np.floor(array_int).astype(np.int64) 
        else:
            raise ValueError(f"invaild rounding method: {rounding}")
        # overflow method
        if overflow == 'wrap': # worst 3n, best 2n
            m = (1<<n_word)
            array_int &= (m-1)
            if signed:
                array_int[array_int>=(1<<(n_word-1))] |= (-m)
            array[...] = array_int * (2 ** -n_frac)
        elif overflow == 'saturate': # worst 4n, best 2n 
            array[...] = array_int * (2 ** -n_frac) # TODO: memory assigment to use same memory, is this really worthy? (idea, use different routine for iadd/isub/imul/....)
            array[array>upper] = upper
            array[array<lower] = lower
        else:
            raise ValueError(f"invalid overflow method: {overflow}")
    return array
    # NOTE: array math above can do in place to save memory

class numfi(np.ndarray):
    #region initialization
    def __new__(cls, input_array=[], s=None, w=None, f=None, **kwargs):
        like = input_array if isinstance(input_array, numfi) and kwargs.get('like',None) is None else kwargs.get('like',None)
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
        if self.f >= 64: # NOTE: this will hit np.int64 bound
            raise ValueError("fraction length too large")
    #endregion initialization
    #region property 
    #TODO: s/w/f rounding/overflow read only?
    @property
    def ndarray(self):
        return self.view(np.ndarray)
    @property
    def int(self):
        return (self.ndarray * 2**self.f).astype(np.int64)
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
            func = lambda i: np.base_repr(i,base=base) #TODO: hex and other base_repr should follow bin, with width=xxx
        return np.vectorize(func)(self.int)        
    #endregion methods    
    #region overload operators
    def __repr__(self):
        signed = 's' if self.s else 'u'
        return super().__repr__() + f' {signed}{self.w}/{self.f}-{self.rounding[0]}/{self.overflow[0]}'
    def __getitem__(self,key):
        item = super().__getitem__(key) # return numfi with shape (1,) instead of single value
        return item if isinstance(item, numfi) else numfi(item, like=self)
    def __setitem__(self, key, item):
        quantized = quantize(item, self.s, self.w, self.f, self.rounding, self.overflow)
        super().__setitem__(key,quantized)
    #endregion overload operators
    #region overload arithmetic operators     
    def __fixed_arithmetic__(self, func, y, add_flag):
        y = y if isinstance(y, numfi) else numfi(y, like=self)
        s = 1 if self.s or y.s else 0
        if add_flag:  # add_flag is true, perform add/sub, otherwise mul/div #TODO: more readable and fast method?
            i = max(self.i, y.i)
            f = max(self.f, y.f)            
            result = numfi(func(y), s, i+f+s+1, f, like=self)
        else:
            result = numfi(func(y).view(np.ndarray), s, self.w+y.w, self.f+y.f, like=self)
        if self.fixed: 
            return numfi(result,like=self) # if fixed, quantize full precision result to fixed length
        elif y.fixed:
            return numfi(result,like=y)            
        else: 
            return result

    __add__         = lambda self,y: self.__fixed_arithmetic__(super().__add__,  y, True)
    __radd__        = lambda self,y: self.__fixed_arithmetic__(super().__radd__, y, True)
    __iadd__        = lambda self,y: self.__fixed_arithmetic__(super().__iadd__, y, True)
    __sub__         = lambda self,y: self.__fixed_arithmetic__(super().__sub__,  y, True)
    __rsub__        = lambda self,y: self.__fixed_arithmetic__(super().__rsub__, y, True)
    __isub__        = lambda self,y: self.__fixed_arithmetic__(super().__isub__, y, True)

    __mul__         = lambda self,y: self.__fixed_arithmetic__(super().__mul__,       y, False)
    __rmul__        = lambda self,y: self.__fixed_arithmetic__(super().__rmul__,      y, False)
    __imul__        = lambda self,y: self.__fixed_arithmetic__(super().__imul__,      y, False)
    __truediv__     = lambda self,y: self.__fixed_arithmetic__(super().__truediv__,   y, False)
    __rtruediv__    = lambda self,y: self.__fixed_arithmetic__(super().__rtruediv__,  y, False)
    __itruediv__    = lambda self,y: self.__fixed_arithmetic__(super().__itruediv__,  y, False)
    __floordiv__    = lambda self,y: self.__fixed_arithmetic__(super().__floordiv__,  y, False)
    __rfloordiv__   = lambda self,y: self.__fixed_arithmetic__(super().__rfloordiv__, y, False)
    __ifloordiv__   = lambda self,y: self.__fixed_arithmetic__(super().__ifloordiv__, y, False)

    __neg__         = lambda self:   numfi(-self.ndarray, like=self) # TODO: when self is unsigned, __neg__ is all zero or invalid? (in matlab it's all zero)
    __invert__      = lambda self:   numfi(~self.ndarray, like=self)
    __pow__         = lambda self,y: numfi(self.ndarray ** y, like=self) # TODO: should n_word/n_frac change in __pow__?
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