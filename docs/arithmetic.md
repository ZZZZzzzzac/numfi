# Fixed-point arithmetic
```python
x = numfi([1,2,3],1,16,8)
# By overloading operators, numfi object can do fixed-point arithmetic easily:
y = x + 1
y = [1] - x
y = x * np.random.rand(3)
y = numfi([1,0,0.1234],1,21,15) / x
y = x ** 0.5
y = x % 3
y = x << 4
...

# By inheriting from numpy.ndarray, numfi object can be used just like normal numpy array, and return same numfi object back
y = np.sin(x)
y = x[x>1]
y = x.sum()
y = x.reshape(3,1)
plt.plot(x)
pandas.DataFrame(x)
for i in x:
    print(i)
numpy.convolve(x,np.ones(4))
numpy.fft.fft(x,n=512)
```

## How numfi do fixed-point arithmetic
numfi use np.float64 as underlying data type, not the 'integer bits' that fixed-point arithmetic usually used, so numfi's 'fixed-point arithmetic' is actually happened in floating-point domain with some precision/quantization/overflow process.  Since it is actually floating-point arithmetic, under some extreme condition the calculation result may not be same as real fixed-point arithmetic using integer representation. (although this is mostly unlikely)

Speed is the reason why numfi use floating-point arithmetic to 'simulate' fixed-point arithmetic:  

1. for modern desktop CPU there is no significant preformance differece between floating-point and integer arithmetic. so floating-point calculation is not as 'slow' as embedded chip  

2. there are some extra step for integer representation during calculation, like to align two fixed-point number with different fraction length we need shift one oprands then do addidtion/subtraction  

3. some operations other than fixed-point arithmetic, like `np.sin()`\advanced indexing\etc.., will take its underlying data as input. If we store integer we have to convert to floating value first for every function, which is inconvient and costly.  

