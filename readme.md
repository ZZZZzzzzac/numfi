# numfi
numfi is a numpy.ndarray subclass that does fixed-point arithmetic.

Feature:
- Automatically perform fixed-point arithmetic through overloaded operators
- As a numpy.ndarray subclass, it's compatible with numpy and other library, just like a normal numpy.ndarray
- Minimize quantization as much as possible to achieve maximum calculation speed.

## Install
---
**Prerequisite**: python3 and numpy

currently numfi is not ready for pypi, so you can clone this repo and build/install locally by:
```
python setup.py bdist_wheel

python -m pip install numfi --upgrade --force-reinstall --find-links=./dist 
```
## Quick start
---
```python
from numfi import numfi

# numfi(array, signed, bits_word, bits_frac, rounding='round', overflow='wrap')
x = numfi([1,2,3],1,16,8) 

# any arithmetic operation with numfi will return a numfi object with proper precision and value
y = x + 1 
z = x * 2
w = x / (0.1 + x)
...
```
## Document
---
Details can be found here: [https://numfi.readthedocs.io/en/latest/?](https://numfi.readthedocs.io/en/latest/?)
## License
The project is licensed under the MIT license.