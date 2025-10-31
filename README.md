# Hypoexponential Distribution
See: https://en.wikipedia.org/wiki/Hypoexponential_distribution

## Implementation
The `hypoexpon_gen` class inherits from `scipy.stats.rv_continuous`, so all standard methods provided by `rv_continuous` are available.  
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

## Usage
```python
from hypoexponential import hypoexpon_gen

# Sum of exponential distributions with means 1, 2, and 3
hypoexpon = hypoexpon_gen(scales=[1.0, 2.0, 3.0])
print(hypoexpon.cdf([0, 2, 4, 6, 8, 10]))

# Duplicate scale parameters are allowed
hypoexpon = hypoexpon_gen(scales=[1.0, 1.0, 0.5])
print(hypoexpon.cdf([0, 2, 4, 6, 8, 10]))
```
