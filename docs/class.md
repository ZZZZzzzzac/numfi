# class numfi(numpy.ndarray)

This class is inherited from `numpy.ndarray`, and add some attributes and methods to support fixed-point arithmetic. It's properties and methods are mimic Matlab's `fi` class, make porting code between Matlab and Python easier.(An easy trick is `import numfi as fi`)

## Create new numfi object

```python
numfi(array=[], s=1, w=16, f=None, RoundingMethod='Nearest', OverflowAction='Saturate', FullPrecision=True, like=None)
```

- `array`: *(`int`/`float`/`list`/`numpy.ndarray`/`numfi`)*, default:`[]`  

    Create a numfi object based on `array`. `array` will be passed to `np.asarray`, so it can be any vaild input of `np.asarray`.  
    If `array` is numfi object and `like` is not defined, will use `array` as template `like`

- `s`: *any*, default:`1`  

    signed or not, will be evaluate as a 0 for unsigned or 1 for signed

- `w`: *`int`*, default:`16`  

    bits of word, must be positive integer

- `f`: *`int`*, default:`None`  

    bits of fraction. If `f=None`, `numfi` will use the maximal percision that fit `array`. (which means allocating as many bits as possible to fraction bits of entire word without overflow, this may lead to negative `f` and `i`).
    Negative `f` and `i` are also supported like Matlab, see [fixed-point arithmetic](arithmetic.md) for details.

- `RoundingMethod`: `str`, default:`'Nearest'`  

    How to round floating point value to integer, method name is same as Matlab's.

  - `'Nearest', 'Round', 'Convergent'`: use np.round(), round towards nearest integer 
  - `'Floor'`: use np.floor(), round towards negative infinity  
  - `'Ceiling'`: use np.ceil(), round towards to positive infinity
  - `'Zero'`: use astype(np.int64), round towards zero  
&nbsp;&nbsp;  

- `OverflowAction`: `str`, default:`'Saturate'`  

    How to deal with overflow during quantization.

  - `'Wrap'`: overflow/underflow will wrap to opposite side
  - `'Saturate'`: overflow/underflow will saturate at max/min value possible
  - `'Error'`: raise OverflowError when overflow happens.
&nbsp;&nbsp;  

- `like`: `numfi / None`, default:`None`

    create new numfi from template `like`. if both keywords arguments and template `like` are given, new argument will have following priority:  **keywords > template(like) > default**

- `FullPrecision`: bool, default:`True`

    if `FullPrecision` is `False`, word bits and fraction bits will not grow during fixed-point arithmetic, otherwise both will grow to keep full result precision. For details see [fixed-point arithmetic](arithmetic.md)

Example:  

```python
x = numfi(1)
x = numfi([1,2,3],1,16,8)
x = numfi(np.arange(100),0,22,11,RoundingMethod='Floor',OverflowAction='Saturate')
y = numfi(np.zeros((3,3)),like=x)
```

## numfi class properities

- `s, w, f, RoundingMethod, OverflowAction, FullPrecision`  
correspoding attributes in numfi object creation. See above for details.  
*Note: these properities are read only, creat new numfi object with new properities if you need change them*

- `i`
the integer bits of numfi object, `i = w - s - f`.

- `ndarray`  
the "raw" form of a numfi object, and the content stored in memory.
*numfi.ndarray is a 'view' of itself with type numpy.ndarray*

- `int`  
the integer representation of numfi object, a `numpy.ndarray` with `dtype=np.int64`

- `double` / `data`
the float representation of numfi object, a `numpy.ndarray` with `dtype=np.float64`.
*numfi.int * numfi.precision = numfi.double*

- `bin` / `bin_`  
the binary str representation('0' and '1') of numfi object, a `numpy.ndarray` with `dtype=str`,  
`numpy.bin_` add additional radix point between integer and fraction bits, and 'x' for placeholder if needed(when `f < 0` or `i < 0`).

- `hex`  / `oct` / `dec`
the hex/oct/dec str representation of numfi object.

- `upper` / `lower`  
the upper / lower bound of numfi object based on currently word/fraction bits setting

- `precision`  
the smallest step of numfi object, equal to `2**-f`

## numfi class method

- `base_repr(self, base=2, frac_point=False)`  
convert numfi's integer representation(numfi.int) to `base` string representation. `numfi.bin/bin_/hex/oct/dec` call this method

### Static Method

- `do_rounding(iarray, RoundingMethod)`
Round `iarray` to integer with various method, see `RoundingMethod` above.
*`iarray` is equal to `float_array * 2**f`, means integer repesenatation of `float_array`*

- `do_overflow(iarray, s, w, f, OverflowAction)`
Do `OverflowAction` if `iarray` overflow, see `OverflowAction` above.

- `get_best_precision(x, s, w)`
Find maximal fraction bits for given data `x` with certain bit width `s` and `w`.
