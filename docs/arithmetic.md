# Overloaded arithmetic operators
## basic arithmetic
```python 
z = x + 1
z = [1] - x
z = x * np.random.rand(3)
z = numfi([1,0,0.1234],1,21,15) / x
```
For basic arithmetic operators of numfi object `x`, if secondary operand `y` is not numfi, it will first convert `y` to numfi with same setting as `x` then do the rest arithmetic.  

If one of the operands, `x` or `y` has `fixed=True`, then the result will keep same quantization setting unchanged, otherwise it will follow rules below:  (*In any case, rounding/overflow/fixed will keep unchanged*)

### ADD/SUB
```python
new_s = 1 if x.s or y.s else 0
new_w = max(x.i,y.i) + max(x.f,y.f) + new_s + 1
new_f = max(x.f,y.f)
```
### MUL
```python
new_s = 1 if x.s or y.s else 0
new_w = x.w + y.w
new_f = x.f + y.f
```

### DIV
```python
new_s = 1 if x.s or y.s else 0
new_w = max(x.w, y.w)
new_f = x.f - y.f
```
*division is the most complicit part, in practice we usually avoid division and use multiply instead, and the full precision result usually cannot be represented by fixed-point*


## other arithmetic
```python
y = -x          <==>  numfi(-x.ndarray, like=x)
y = ~x          # invert only works on integer, for fixed-point it is same as `-x`
y = x ** 0.5    <==>  numfi(x.ndarray ** 0.5, like=x)
y = x % 3       <==>  numfi(x.ndarray % 3, like=x)
```
arithmetic other than add/sub/mul/div will use `numfi.ndarray` and keep quantization setting unchanged.

## bitwise operation
```python
y = x & 0b101   <==>  numfi((x.int & 0b101) * x.precision, like=x) 
y = x | 0b100   <==>  numfi((x.int | 0b100) * x.precision, like=x) 
y = x ^ 0b001   <==>  numfi((x.int ^ 0b001) * x.precision, like=x) 
y = x << 4      <==>  numfi((x.int << 4) * x.precision, like=x) 
y = x >> 2      <==>  numfi((x.int >> 2) * x.precision, like=x) 
```
these bitwise operators will use `numfi.int` to do bitwise operation, then convert back to numfi object

## logical comparsion
```python
y = x > 0.5           <==>   x.ndarray > 0.5
y = x >= 0.5          <==>   x.ndarray >= 0.5
y = x == x            <==>   x.ndarray == x
y = x != [0,0,0]      <==>   x.ndarray != [0,0,0]
y = x <= np.ones(3)   <==>   x.ndarray <= np.ones(3)
y = x < [1,1,1]       <==>   x.ndarray < [1,1,1]
``` 
logical comparsion operators will use `numfi.ndarray` to compare with operand.


# How numfi do fixed-point arithmetic
In theory, to accurately simulate fixed-point arithmetic bit by bit, we should store integers in memory and perform the fixed-point arithmetic in the integer domain. However there are some problems with this approach:

1. there may be extra step during arithmetic. For example: `a(s16/8) + b(s16/10)`, to add them together in integer domain we must first left shift `a` by 2 to align the radix point.

2. not compatible with other function like `np.sin(x)/plt.plot(x)/etc`. They will use data in memory, which is the integer representation instead of 'real-value' floating-point data. We have to write extra code like `np.sin(x.float)` to get correct answer, this not only makes porting code between floating and fixed point harder, but also introduce extra computation.

numfi use 'real-value' np.float64 as underlying data type instead of integer, so numfi's 'fixed-point arithmetic' is actually happened in floating-point domain with some quantization/overflow control. Since it's essentially floating-point arithmetic, under some extreme condition the result may not be exactly the same as real fixed-point arithmetic using integer representation. (due to precision limit of floating point data type)

Speed and compatibility is the reason why numfi use floating-point arithmetic to 'simulate' fixed-point arithmetic. Unlike embedded chips, for modern desktop CPU there is no significant preformance differece between floating-point and integer arithmetic. And as a `numpy.ndarray` subclass, numfi object using 'real-value' floating-point data can be used in any floating-point algorithm flawlessly, take the advantage of python's dynamic type feature

