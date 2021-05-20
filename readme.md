# [numfi](https://github.com/ZZZZzzzzac/numfi)
numfi is a numpy.ndarray subclass that does fixed-point arithmetic.

Feature:  

- Automatically perform fixed-point arithmetic through overloaded operators  

- Maximum compatibility with numpy and other library, just like a normal numpy.ndarray  

- Optimized calculation speed by minimizing quantization as much as possible   

## Install
**Prerequisite**: python3 and numpy

```
pip install numfi
```
or you can just copy [numfi.py](https://github.com/ZZZZzzzzac/numfi/blob/master/numfi/numfi.py) and do whatever you want, after all it's only 200 lines of code

## Quick start
```python
from numfi import numfi

# numfi(array=[], signed=1, bits_word=32, bits_frac=16, rounding='round', overflow='wrap')
x = numfi(np.random.rand(3,3),1,16,8) 

# any arithmetic operation with numfi will return a numfi object with proper precision and value
y = x + 1 
z = x * 2
w = np.sin(x)
...
```
## Document
Details can be found here: [https://numfi.readthedocs.io/en/latest/?](https://numfi.readthedocs.io/en/latest/?)

## License
The project is licensed under the MIT license.