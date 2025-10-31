# Hypoexponential distribution
See https://en.wikipedia.org/wiki/Hypoexponential_distribution

# Usage
```python
from hypoexponential import hypoexpon_gen

# Sum of exponential distributions whose means are 1, 2, and 3
hypoexpon = hypoexpon_gen(scales=[1.0, 2.0, 3.0])
print(hypoexpon.cdf([0, 2, 4, 6, 8, 10]))

# Accepts scale parameters with duplicate values
hypoexpon = hypoexpon_gen(scales=[1.0, 1.0, 0.5])
print(hypoexpon.cdf([0, 2, 4, 6, 8, 10]))
```
