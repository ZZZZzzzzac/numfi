# What is fixed-point and how to convert float-point to fixed-point

In short, numfi use [matlab's definition](https://www.mathworks.com/help/fixedpoint/ug/scaling.html) of fixed-point number, with `scale=1` and `bias=0`.

## Detail of fixed-point format

To convert a float-point number to fixed-point, we use following equation:  
`fixed = (scale*2^fraction_bits) * float + bias`  
 where `scale=1`, `bias=0` and `fraction_bits=f` in numfi, it becomes: `fixed = 2^f * float` or `float = 2^-f * fixed`.

For example, to convert the floating-point number `3.1415926` to a fixed-point number with 6 fraction bits, it becomes `3.1415926 * 2^6 = 201.0619264`. The integer `201` is the fixed-point representation of `3.1415926` with 6 fraction bits. This process is called ***quantization***.

## Quantize error

You may notice that `201/2^6 = 3.140625`, which is not exactly `3.1415926`, this difference is called ***quantization error***. The quantization error is the difference between the actual value and the quantized value, and it depends on length of fraction bit. More fraction bits means better precision, but in most cases quantize error will never be zero.

## fixed-point range

We want as many fraction bits as possible for better precision, but in real life we don't have infinite storage. Total bits we can use to represent a fixed-point number is called ***word length***, which is `w` in numfi. And with a fixed word length, there is a limited range of integer it can represent, which also means same for its float-point counterpart.

The range can be calculated using the following equations: `[-2^i, 2^i-2^-f]` for signed numbers and `[0, 2^i-2^-f]` for unsigned numbers. Exceeding this range will result in ***overflow***.

For example, if we use `w=8`, the maximum unsigned integer for 8 bits is `0xff`. With `f=6`, the maximum floating-point value it can represent is `0xff * 2^-6 = 3.984375`, and the minimum is `0`. This fixed-point format has a range of `[0, 3.984375]`. If we use a signed integer, the range becomes `[-2, 1.984375]`.

We use signed bit `s=1/0` to represent whether the integer is signed or unsigned, and integer bits `i=w-f-s` to represent how many bits are used to represent integer part. Like previous example, we have `w=8,f=6,s=1`, then `i=8-6-1=1`, then range is `[-2^i,2^i-2^-f] = [-2,1.984375]`.

## negative f and i

One confusing aspect is that `f` or `i` can be negative, which implies `i>w` or `f>w`. To understand this, one should first learn about `s/w/f/i` in binary form:

Say if we have infinite word and fraction bits to represent a floating-point number, with value `3.4` we have:
(we use `RoundingMethod="Zero"` here to avoid rounding process, and assume no overflow)

```python
fi(3.4, 1, inf, inf).bin_ = ...00000011.011001100110011001100110... #(numfi can't really take `inf` as argument, just a demonstration)
```

When we use finite length fixed point like `s=1,w=8,f=4,i=3`, actually we just take a piece of that infinite binary sequence above.

with both `f>0` and `i>0`, it is easy to understand that `i` is how many bits on the left side of radix point(exclude signed bit if exist), `f` is how many bits on the right side of radix point. The LSB of word bits is what we called `precision`, because it is the minimal step of fixed-point number and equal to `2^-f`

And you can see that, we can not include all bits inside an 8 bit word integer, those lost fraction bits are what ***quantization error*** looks like.

```python
 ...0000|0011.0110|01100110011001100110...
        |< w = 8 >| = fi(3.4, s=1, w=8, f=4).bin_ = 

   2^3     2^2     2^1     2^0   . 2^-1     2^-2     2^-3     2^-4
    0       0       1       1    .  0        1        1        0   
= (2^3*0 + 2^2*1 + 2^1*1 + 2^0*1 + 2^-1*0 + 2^-2*1 + 2^-3*1 + 2^-4*0)
= ( 0    +  0    +  2    +  1    +  0     +  0.25  +  0.125 +  0    ) = 3.375
```

when we change `f` or `i`, we actually just pick a different piece from that infinite bits sequence. For example here is the case of `f=0`:

```python
...|00000011|.011001100110011001100110...
   |< w = 8>| = fi(3.4, s=1, w=8, f=0).bin_ = 

   2^7     2^6     2^5     2^4     2^3     2^2     2^1     2^0  .
    0       0       0       0       0       0       1       1   .     
= (2^7*0 + 2^6*0 + 2^5*0 + 2^4*0 + 2^3*0 + 2^2*0 + 2^1*1 + 2^0*1)
= ( 0    +  0    +  0    +  0    +  0    +  0    +  2    +  1   ) = 3
```

all fraction bits after radix point are lost, the float-point number of this fixed-point format has no fraction part. LSB is first bit left to radix point, which is `2^-f=2^0=1`

Now `f=-1` is easy to understand: move the start point of 8 bit piece to left 1 more bit, LSB is second bit left to radix point, which is `2^-f=2^-(-1)=2`

letter `x` is a placeholder that represent a lost bit, so that you can see where is the radix point. All lost bits have default value of `0`.

```python
...|00000001|1.011001100110011001100110...
   |< w = 8>| = fi(3.4, s=1, w=8, f=-1).bin_ = 

   2^8     2^7     2^6     2^5     2^4     2^3     2^2     2^1     2^0  .
    0       0       0       0       0       0       0       1       x   .   
= (2^8*0 + 2^7*0 + 2^6*0 + 2^5*0 + 2^4*0 + 2^3*0 + 2^2*1 + 2^1*0)
= ( 0    +  0    +  0    +  0    +  0    +  0    +  0    +  2   ) = 2
```

same for negative `i`, means the start point of piece is right to radix point. Of course with example above, when you do that you will lost bits in integer part and that is what ***overflow*** looks like. So here we use `0.2` as example:

```python
fi(0.2, 1, inf, inf).bin_ = 
...0000000|0.0011001|10011001100110011...
          |< w = 8 >| = fi(0.2, 1, 8, 7) = 

   2^0  .  2^-1     2^-2     2^-3     2^-4     2^-5     2^-6     2^-7
    0   .   0        0        1        1        0        0        1   
= (2^0*0 + 2^-1*1 + 2^-2*1 + 2^-3*1 + 2^-4*0 + 2^-5*1 + 2^-6*1 + 2^-7*1)
= ( 0    +  0     +  0     +  0.125 + 0.0625 +  0     +  0     +0.0078125) = 0.1953125
```

```python
...00000000.0|01100110|011001100110011...
             |< w = 8>| = fi(0.2, 1, 8, 9) = 

      .   2^-1     2^-2     2^-3     2^-4     2^-5     2^-6     2^-7      2^-8       2^-9
      .    x        0        1        1        0        0        1         1          0
=                 (2^-2*0 + 2^-3*1 + 2^-4*1 + 2^-5*1 + 2^-6*0 + 2^-7*1  + 2^-8*1   + 2^-9*1)
=                 ( 0     +  0.125 + 0.0625 +  0     +  0     +0.0078125+0.00390625+  0    ) = 0.19921875
```

## negative value

We have two's complement to represent negative integer number, and using `float = 2^-f * fixed` we can also represent negative floating-point number.
But what does it look like in binary form? Answer is simple: binary weight of sign bit is negative when sign bit is `1`, and rest bits are the same as above.

```python
fi(-3.4, 1, inf, inf).bin_ = 
  ...111|11100.100|110011001100110011010...
        |< w = 8 >| = fi(-3.4, 1, 8, 3, RoundingMethod="Floor").bin = 

   2^4     2^3     2^2     2^1     2^0   . 2^-1     2^-2     2^-3
    1       1       1       0       0    .  1        0        0   
= (2^4*0 + 2^3*1 + 2^2*0 + 2^1*0 + 2^0*0 + 2^-1*1 + 2^-2*1 + 2^-3*1)
= (-16   +  8    +  4    +  0    +  0    +  0.5   +  0     +  0    ) = -3.5
   ^ negative 
```
