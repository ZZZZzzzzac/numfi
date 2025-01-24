# class numfi

This class is inherited from meta class `numfi_tmp`, add some attributes and methods to support fixed-point arithmetic. Its properties and methods mimic Matlab's `fi` class, making it easier to port code between Matlab and Python.(An easy trick is `from numfi import numfi as fi`)

## Create new numfi object

```python
numfi(array, s, w, f, RoundingMethod='Nearest', OverflowAction='Saturate', FullPrecision=True, quantize=True, iscpx=False, like=None)
```

- `array`: *(`int`/`float`/`list`/`numpy.ndarray`/`numfi`)*, default:`[]`  
    Create a numfi object based on `array`. `array` will be passed to `np.asarray`, so it can be any valid input for `np.asarray`.  
    If `array` is a `numfi` object and `like` is not defined, `array` will be used as the template for `like`. See the `like` argument below for details.

- `s`: *`int`*, default:`1`  
    `1` for signed, `0` for unsigned.

- `w`: *`int`*, default:`16`  
    bits of word, must be positive integer.

- `f`: *`int`*, default:`None`  
    bits of fraction. If `f=None`, `numfi` will use the maximal precision that fit `array`. (which means allocating as many bits as possible to fraction bits of entire word without overflow, this may lead to negative `f` and `i`). Negative `f` and `i` are also supported, similar to Matlab. See [this](fixed_point.md) for details.

- `RoundingMethod`: `str`, default:`'Nearest'`  
    How to round floating point value to integer, method name is same as Matlab's.
  - `'Nearest', 'Round', 'Convergent'`: use np.round(), round towards nearest integer
  - `'Floor'`: use np.floor(), round towards negative infinity  
  - `'Ceiling'`: use np.ceil(), round towards to positive infinity
  - `'Zero'`: use astype(np.int64), round towards zero  
&nbsp;&nbsp;  

- `OverflowAction`: `str`, default:`'Saturate'`  
    How to deal with overflow during quantization.
  - `'Wrap'`: overflow/underflow will wrap to opposite side, just like binary integer.
  - `'Saturate'`: overflow/underflow will saturate at max/min value possible
  - `'Error'`: raise OverflowError when overflow happens.
  - `'warning'`: raise Warning when overflow happens.
  - `'Ignore'`: ignore overflow/underflow, keep the value as is.
&nbsp;&nbsp;  

- `FullPrecision`: bool, default:`True`  
    If `FullPrecision` is `False`, the word bits and fraction bits will not grow during fixed-point arithmetic. Otherwise, both will grow to maintain full result precision.  
    For details see [fixed-point arithmetic](arithmetic.md)

- `quantize`: bool, default:`True`  
    If `quantize` is `False`, numfi will take the value of `array` "as is" without quantization, but will truncate bits that exceed `w`. i.e.: `numfi(123,1,8,4,quantize=False).int == 123`

- `iscpx`: bool, default:`False`  
    When `array` is not complex but `iscpx=True`, a zero imaginary part will be added. Otherwise `iscpx` is dependent on whether `array` is complex or not.

- `like`: `numfi / None`, default:`None`  
    Create a new numfi object from the template `like`. If both keyword arguments and the template `like` are provided, the new arguments will have the following priority:  
    **keywords > template(like) > default**.

Initialize numfi example:

```python
x = numfi(1)
x = numfi([1, 2, 3], 1, 16, 8)
x = numfi(np.arange(100), 0, 22, 11, RoundingMethod='Floor', OverflowAction='Saturate')
y = numfi(np.zeros((3, 3)), like=x)
```

## numfi class properties  

*Note: these properties are read only, create new numfi object with new properties if you need change them*

- `s, w, f, RoundingMethod, OverflowAction, FullPrecision, iscpx`  
    Corresponding attributes in numfi object creation. See above for details.  

- `i`  
    The integer bits of numfi object, `i = w - s - f`.

- `ndarray`  
    The 'raw' form of a numfi object, which is what is actually stored in memory.
    *numfi.ndarray is a 'view' of itself with type numpy.ndarray*

- `int`  
    The integer representation of the numfi object, a `numpy.ndarray` with `dtype=np.int64`.

- `double` / `data`  
    The floating-point representation of the `numfi` object, a `numpy.ndarray` with `dtype=np.float64`.  
    *numfi.int * numfi.precision = numfi.double*

- `bin` / `bin_`  
    The binary string representation ('0' and '1') of the `numfi` object, a `numpy.ndarray` with `dtype=str`.  
    `numpy.bin_` add additional radix point between integer and fraction bits, and 'x' for placeholder if needed(when `f < 0` or `i < 0`).

- `hex`  / `oct` / `dec`  
    The hex/oct/dec str representation of numfi object.

- `upper` / `lower`  
    The upper and lower bounds of the `numfi` object based on the current word and fraction bits settings: `upper = (2**i) - 2**-f`, `lower = -(2**i) if s is true, otherwise 0`.

- `precision`  
    The smallest step (1 in integer form) of the `numfi` object, equal to `2**-f`.

- `Value`  
    The string representation of the `numfi` object.

## numfi class method

- `base_repr(self, base=2, frac_point=False)`  
    Convert numfi's integer representation(numfi.int) to `base` string representation. `numfi.bin/bin_/hex/oct/dec` call this method.

### Static Method

- `do_rounding(iarray, RoundingMethod)`  
    Round `iarray` to integer with various method, see `RoundingMethod` above.  
    *`iarray` is integer representation of `array`, equals to `array * 2**f`*

- `do_overflow(iarray, s, w, f, OverflowAction)`  
    Do `OverflowAction` if `iarray` overflow, see `OverflowAction` above.  

- `get_best_precision(x, s, w)`  
    Find maximal fraction bits for given data `x` with certain bit width `s` and `w`.  

- `quantize(array, s, w, f, RoundingMethod, OverflowAction, quantize=True)`  
    Main function of numfi to quantize float array to fixed-point integer array with arguments mentioned above. Call `do_rounding` and `do_overflow` inside.  

## class numqi

There is another class `numqi` which is similar to `numfi`, the only difference is, `numqi` store fixed-point integer in memory, but `numfi` store its float-point number. Everything else is the same.
