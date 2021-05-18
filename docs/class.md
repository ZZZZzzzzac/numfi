# class numfi(numpy.ndarray)

## Create new numfi object: 

```python
numfi(array=[], s=None, w=None, f=None, rounding='round', overflow='saturate', like=None, fixed=False)
```

- `array`: *(`int`/`float`/`list`/`numpy.ndarray`/`numfi`)*, default:`[]`  

    Create a numfi object based on `array`. `array` can be single value (`int`/`float`) or multidimension array (list of numbers or numpy.ndarray).  
    If `array` is numfi object and `like` is not defined, will use `array` as template `like`

- `s`: *any*, default:`1`  

    signed or not, will be evaluate as a boolean value  

- `w`: *`int`*, default:`32`  

    bits of word

- `f`: *`int`*, default:`16`  

    bits of fraction. Note that unlike matlab, `f` must be less or equal to `w`, means negative integer bits is not supported in numfi

- `rounding`: `str`, default:`'round'`  

    one of `'round'`/`'floor'`, used during quantization    

    - `'round'`: use np.round(), round to nearest integer 
    - `'floor'`: use np.floor(), round to negative infinity  
  

- `overflow`: `str`, default:`'saturate'`  

    one of `'wrap'`/`'saturate'`, used during quantization

    - `'wrap'`: overflow/underflow will wrap to opposite side
    - `'saturate'`: overflow/underflow will saturate at max/min value possible

- `like`: `numfi`/None, default:`None`

    create new numfi from template `like`. if both keywords arguments and template `like` are given, new argument will have following priority:  **keywords > template(like) > default**

- `fixed`: bool, default:`False`

    if `fixed` is `True`, word bits and fraction bits will not grow during fixed-point arithmetic, otherwise both will grow to keep full result precision. For details see [fixed-point arithmetic]()

Example:
```python
x = numfi(1)
x = numfi([1,2,3],1,16,8)
x = numfi(np.arange(100),0,22,11,rounding='floor',overflow='saturate')
y = numfi(np.zeros((3,3)),like=x)
```

## numfi properities

- `numfi.s, numfi.w, numfi.f, numfi.rounding, numfi.overflow, numfi.fixed`  
correspoding attributes in numfi object creation. See above for details.  
*Note: these properities should be read only but not protected in code for simplicity. To change these properities one should use numfi creation to create new object*
```python
x = numfi([1,2,3],1,16,8)
new_x = numfi(x,s=0,like=x)
```

- `numfi.ndarray`  
the underlying data of numfi object. a `numpy.ndarray` with `dtype=np.float64`

- `numfi.int`  
the integer representation of numfi object, a `numpy.ndarray` with `dtype=np.int64`

- `numfi.bin` / `numfi.bin_`  
the binary str representation('0' and '1') of numfi object, a `numpy.ndarray` with `dtype=str`,  
`numpy.bin_` has additional radix point between integer and fraction bits

- `numfi.hex`  
the hexadecimal str representation of numfi object.

- `numfi.i`   
the integer bits of numfi object, equal to `numfi.w-numfi.f`

- `numfi.upper` / `numfi.lower`  
the upper / lower bound of numfi object based on currently word/fraction bits setting

- `numfi.precision`  
the smallest step of numfi object, equal to `2**-numfi.f`

