# Overloaded arithmetic operators

## basic arithmetic

```python 
z = x + 1
z = [1] - x
z = x * np.random.rand(3)
z = numfi([1,0,0.1234],1,21,15) / x
```

For basic arithmetic operators of numfi object `x`, if secondary operand `y` is not numfi, it will first convert `y` to numfi with template `x` then do the rest arithmetic.

It will first use `numfi.double` to do 64 bit floating point arithmetic, then convert back to `numfi` with proper `s/w/f`(new s/w/f follow some rules), i.e.:

`y = x + 1  <==>  y = numfi(x.double + numfi(1,like=x).double, s, w, f)`

If one of the operands, `x` or `y` has `FullPrecision=False`, then the result will keep old `s/w/f` unchanged, otherwise it will follow new s/w/f rules below.  (*In any case, `RoundingMethod` and `OverflowAction` will keep unchanged*)

## New s/w/f rules

### Signed/Unsigned

`s = 1 if x.s or y.s else 0`
Answer is unsigned(s=0) if and only if each operand is unsigned. Otherwise, answer is signed.

### ADD/SUB

```python
f = max(x.f, y.f)
w = max(x.i, y.i) + f + s + (1 if (x.s==y.s) else 2) # need extra bit to convert unsigned to signed
```

### MUL

```python
w = x.w + y.w
f = x.f + y.f
```

### DIV

```python
w = max(x.w, y.w)
f = x.f - y.f
```

Division in fixed point arithmetic has many different definition, here we use the same definition as Matlab.
Other division definition will be added in future version.

## other arithmetic

```python
y = -x          <==>  numfi(-x.double,          like=x)
y = x ** 0.5    <==>  numfi(x.double ** 0.5,    like=x)
y = x % 3       <==>  numfi(x.double % 3,       like=x)
```

arithmetic operators other than `add/sub/mul/div` will use `numfi.double` as operand.

## bitwise operation

```python
y = ~x          <==>  numfi((~x.int)        * x.precision, like=x)
y = x & 0b101   <==>  numfi((x.int & 0b101) * x.precision, like=x) 
y = x | 0b100   <==>  numfi((x.int | 0b100) * x.precision, like=x) 
y = x ^ 0b001   <==>  numfi((x.int ^ 0b001) * x.precision, like=x) 
y = x << 4      <==>  numfi((x.int << 4)    * x.precision, like=x) 
y = x >> 2      <==>  numfi((x.int >> 2)    * x.precision, like=x) 
```

Bitwise operators will use `numfi.int` as operand, then convert back to float value then numfi object

## logical comparsion

```python
y = x > 0.5           <==>   x.double > 0.5
y = x >= 0.5          <==>   x.double >= 0.5
y = x == x            <==>   x.double == x
y = x != [0,0,0]      <==>   x.double != [0,0,0]
y = x <= np.ones(3)   <==>   x.double <= np.ones(3)
y = x < [1,1,1]       <==>   x.double < [1,1,1]
```

Logical comparsion operators will use `numfi.double` as operand, result is `numpy.ndarray(dtype=bool)`

## numpy.ndarray ufunc function

```python
y = np.sin(x)   <==>  numfi(np.sin(x.double))
y = np.sum(x)   <==>  numfi(np.sum(x.double))
y = x.mean()    <==>  numfi(x.double.mean())
# y.s == x.s  y.w == x.w
# y.f == numfi.get_best_presicion(y, x.s, x.w) if x.FullPrecision else x.f
```

ufunc function will use `numfi.double` as operand, then convert to `numfi` object. If `FullPrecision=True`, it will find best fraction precision with floating point result, if not, it will keep same s/w/f as operand.

Note this new s/w/f behavior is different from Matlab's, since in Matlab each ufunc has it own `embedded@fi` version, which is not possible in python/numpy.

Example: `np.cos(numfi(0))` is `s16/14` and equal to `1` exactly, which need two integer bit to represent. (remember upper limit of one integer bit is `1 - 2**-f`, not `1`)

But in Matlab `cos(fi(0))` is `s16/15` and equal to `0.999969482421875`. Because Matlab's `cos` function `R2023a\toolbox\fixedpoint\fixedpoint\+embedded\@fi\cos.m` explicitly set answer to `s16/15`. Which make sense since `-1<=cos<=1`, but sacrifice some precision when answer is exactly 1.

## How numfi do fixed-point arithmetic

In theory, to accurately simulate fixed-point arithmetic bit by bit, we should store integers in memory and perform the fixed-point arithmetic in the integer domain. However there are some problems with this approach:

1. there may be extra step during arithmetic. For example: `a(s16/8) + b(s16/10)`, to add them together in integer domain we must first left shift `a` by 2 to align the radix point.

2. not compatible with other function like `np.sin(x)/plt.plot(x)/etc`. They will use data in memory, which is the integer representation instead of 'real-value' floating-point data. We have to write extra code like `np.sin(x.double)` to get correct answer, this not only makes porting code between floating and fixed point harder, but also introduce extra computation.

numfi use 'real-value' `np.float64` as underlying data type instead of integer, so numfi's 'fixed-point arithmetic' is actually happened in floating-point domain with some quantization/overflow control. Since it's essentially floating-point arithmetic, under some extreme condition the result may not be exactly the same as real fixed-point arithmetic using integer representation. (due to precision limit of floating point data type)

Speed and compatibility is the reason why numfi use floating-point arithmetic to 'simulate' fixed-point arithmetic. Unlike embedded chips, for modern desktop CPU there is no significant preformance differece between floating-point and integer arithmetic. And as a `numpy.ndarray` subclass, numfi object using 'real-value' floating-point data can be used in any floating-point algorithm flawlessly, take the advantage of python's dynamic type feature

