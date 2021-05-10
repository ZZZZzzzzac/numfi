#########################
# numfi - a fixed-point class (mimic from matlab's fixed-point object <fi>) inherited from numpy.ndarray
# author : ZZZZzzzzac zinger.kyon@gmail.com
# version : 0.1
#########################

import numpy as np

class numfi(np.ndarray):
    def __new__(cls, input_array, signed=1, w=32, f=16, like=None, quantize=True, **kwargs): 
        ndarray = np.asarray(input_array).astype(float).view(cls)
        if isinstance(like, numfi):
            ndarray.config(like.kwargs)
            ndarray.resize(like.signed, like.w, like.f, quantize)            
        else:     
            ndarray.config(kwargs)                       
            ndarray.resize(signed, w, f, quantize)            
        return ndarray

    def resize(self, signed, w, f, quantize=True):
        self.signed = bool(signed) # false signed = 0/false/None/''/[]
        self.w = w
        self.f = f
        self.i = self.w-self.f-self.signed
        if self.w < self.f:
            raise ValueError("fraction length > word length is not support")
        self.precision = 2**-self.f
        upper = 2**(self.w - self.f - (1 if self.signed else 0))
        self.upper = upper - self.precision 
        self.lower = -2**(self.w-self.f-1) if self.signed else 0    
        
        if quantize:
            self[...] = self # note this will call `__setitem__`, and then call `self.quantize(self)` inside `__setitem__`
        return self

    def quantize(self, value):
        # quantize 
        b = np.asarray(value) * (2**self.f) # np.asarry(value) is to prevent infinite loop
        if self.rounding == 'nearest':
            store_int = np.round(b).astype(int)
        elif self.rounding == 'floor':
            store_int = np.floor(b).astype(int)
        else:
            raise ValueError(f'not support rounding method {self.rounding}')
        if store_int.shape == ():
            store_int = store_int.reshape(1,) # single value will be convert to (1,) array
        # overflow/underflow
        if self.overflow == 'wrap':
            m = (1 << self.w)
            if self.signed: 
                store_int &= (m - 1)              
                store_int[store_int>=(1 << (self.w-1))] |= (-m)
            else: 
                store_int &= (m - 1) 
            quantized = store_int*self.precision 
        elif self.overflow == 'saturate':            
            quantized = store_int*self.precision
            quantized[quantized>self.upper] = self.upper  # np.clip(self,self.upper,self.lower,out=self) will do the same job, but slower
            quantized[quantized<self.lower] = self.lower            
        else:
            raise ValueError(f'not support overflow method {self.overflow}')
        return quantized

    def config(self, kwargs):
        self.rounding = kwargs.get('rounding', 'nearest')
        self.overflow = kwargs.get('overflow', 'saturate')
        self.fixed = kwargs.get('fixed', False)

    @property
    def ndarray(self):
        return self.view(np.ndarray)
    @property
    def int(self):
        return (self.ndarray*2**self.f).astype(int) # return normal ndarray
    @property
    def kwargs(self):
        return {
            'rounding' : self.rounding,
            'overflow' : self.overflow,
            'fixed': self.fixed
        }

    def base_repr(self, base=2, frac_point=False): # return ndarray with same shape and dtype = '<Uw' where w=self.w
        if base == 2:
            add_point = lambda s: s[:-self.f] + '.' + s[-self.f:] if frac_point else s
            func = lambda i: add_point(np.binary_repr(i,width=self.w))    
        else:
            func = lambda i: np.base_repr(i,base=base)
        return np.vectorize(func)(self.int)
    @property
    def bin(self): 
        return self.base_repr(2)
    @property 
    def bin_(self):
        return self.base_repr(2,frac_point=True)
    @property
    def hex(self):
        return self.base_repr(16)

    # overload operands    
    def __repr__(self):
        signed = 's' if self.signed else 'u'
        return f'numfi-{signed}{self.w}/{self.f}: \n' + self.ndarray.__repr__() 
    __str__ = __repr__
    def __getitem__(self, key):        
        return numfi(super().__getitem__(key), like=self)
    def __setitem__(self, index, value):
        super().__setitem__(index, self.quantize(value))
    def __neg__(self):
        return numfi(super().__neg__(), like=self)
    def __pos__(self):
        return self    
    # arithmetic overloading helper
    def __arithmeticADDSUB__(self, func, y):
        y = y if isinstance(y, numfi) else numfi(y, like=self)
        if self.fixed:
            return numfi(func(y.ndarray),like=self)
        elif y.fixed:
            return numfi(func(y.ndarray),like=y)            
        else:
            i = max(self.i, y.i)
            f = max(self.f, y.f)
            signed = self.signed|y.signed
            return numfi(func(y.ndarray), signed, i+f+signed+1, f, None, False, **self.kwargs)         
    def __arithmeticMULDIV__(self, func, y):
        y = y if isinstance(y, numfi) else numfi(y, like=self)
        if self.fixed:
            return numfi(func(y.ndarray), like=self)
        elif y.fixed:
            return numfi(func(y.ndarray), like=y)
        else:
            return numfi(func(y.ndarray), self.signed|y.signed, self.w+y.w, self.f+y.f, None, False, **self.kwargs) 
    # arithmetic
    __add__         = lambda self,y: self.__arithmeticADDSUB__(super().__add__, y)
    __radd__        = lambda self,y: self.__arithmeticADDSUB__(super().__radd__, y)
    __iadd__        = lambda self,y: self.__arithmeticADDSUB__(super().__iadd__, y)
    __sub__         = lambda self,y: self.__arithmeticADDSUB__(super().__sub__, y)
    __rsub__        = lambda self,y: self.__arithmeticADDSUB__(super().__rsub__, y)
    __isub__        = lambda self,y: self.__arithmeticADDSUB__(super().__isub__, y)
    __mul__         = lambda self,y: self.__arithmeticMULDIV__(super().__mul__, y)
    __rmul__        = lambda self,y: self.__arithmeticMULDIV__(super().__rmul__, y)
    __imul__        = lambda self,y: self.__arithmeticMULDIV__(super().__imul__, y)
    __truediv__     = lambda self,y: self.__arithmeticMULDIV__(super().__truediv__, y)
    __rtruediv__    = lambda self,y: self.__arithmeticMULDIV__(super().__rtruediv__, y)
    __itruediv__    = lambda self,y: self.__arithmeticMULDIV__(super().__itruediv__, y)
    __floordiv__    = lambda self,y: self.__arithmeticMULDIV__(super().__floordiv__, y)
    __rfloordiv__   = lambda self,y: self.__arithmeticMULDIV__(super().__rfloordiv__, y)
    __ifloordiv__   = lambda self,y: self.__arithmeticMULDIV__(super().__ifloordiv__, y)
    #TODO: more advance operation? (shift expand/quantize of logical operation)
    __mod__         = lambda self,y: numfi(super().__mod__(y.ndarray), like=self)
    __lshift__      = lambda self,y: numfi(super().__lshift__(y.ndarray), like=self)
    __rshift__      = lambda self,y: numfi(super().__rshift__(y.ndarray), like=self)
    __and__         = lambda self,y: numfi(super().__and__(y.ndarray), like=self) 
    __or__          = lambda self,y: numfi(super().__or__(y.ndarray), like=self) 
    __xor__         = lambda self,y: numfi(super().__xor__(y.ndarray), like=self) 
    __invert__      = lambda self,y: numfi(super().__invert__(y.ndarray), like=self)
    # comparison
    __eq__          = lambda self,y: self.ndarray == y
    __ne__          = lambda self,y: self.ndarray != y
    __ge__          = lambda self,y: self.ndarray >= y
    __gt__          = lambda self,y: self.ndarray >  y
    __le__          = lambda self,y: self.ndarray <= y
    __lt__          = lambda self,y: self.ndarray <  y



