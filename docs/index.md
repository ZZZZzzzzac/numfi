# [numfi](https://github.com/ZZZZzzzzac/numfi)

numfi is a numpy.ndarray subclass that does fixed-point arithmetic.

Features:  

- Automatically perform fixed-point arithmetic through overloaded operators

- Maximum compatibility with numpy and other library, just like a normal numpy.ndarray  

- Optimized calculation speed, and keep bit precision as much as possible

- mimic the behavior of matlab's fixed-point toolbox

## Install

**Prerequisite**: python3 and numpy

```bash
pip install numfi
```

or you can just copy [numfi.py](https://github.com/ZZZZzzzzac/numfi/blob/master/numfi/numfi.py) and do whatever you want, after all it's only 300 lines of code

## Quick start

```python
from numfi import numfi as fi
import numpy as np

# numfi(array=[], s=1, w=16, f=None, RoundingMethod='Nearest', OverflowAction='Saturate')
x = fi([1,0,0.1234],1,16,8) 
# print(numfi) returns a brief description of the numfi object: x => s16/8-N/S
# s for 'signed', u for 'unsigned', followed by word bits(w=16) and fraction bits(f=8), N/S for 'Nearest' and 'Saturate' for rounding/overflow method

# Any arithmetic operation with numfi will return a numfi object with proper precision and value.
# By overloading operators, numfi objects can perform fixed-point arithmetic easily.

# normal arithmetic operation work with float form of x
y = x + 1
y = [1] - x
y = x * [3,0,-3]
y = fi([1,0,0.1234],1,21,15) / x
y = -x
y = x ** 0.5
y = x % 3
# Comparisons return np.array of bool, just like a normal np.array.
y = x > 0.5
y = x >= fi([1,0,0.1234],1,21,15)
y = x == x
y = x <= np.ones(3)
y = x < [1,1,1]
# Bitwise operations work with the integer form of x.
y = x & 0b101 
y = 0b100 | x   # order of operands doesn't matter
y = x ^ x       # two numfi object can also be used in bitwise operations
y = x << 4
y = x >> 2
...

# numfi objects can be used just like normal numpy arrays and return the same numfi object back.
y = np.sin(x)
y = x[x>1]
y = x.sum()
y = x.reshape(3,1)
np.convolve(x[0],np.ones(3))
np.fft.fft(x,n=512)
plt.plot(x)
pandas.DataFrame(x)
f, t, Sxx = scipy.signal.spectrogram(x,nperseg=256,noverlap=128)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
for i in x:
    print(i)
...
```

## Document

Details can be found here: [https://numfi.readthedocs.io/en/latest](https://numfi.readthedocs.io/en/latest)

## License

The project is licensed under the MIT License.
