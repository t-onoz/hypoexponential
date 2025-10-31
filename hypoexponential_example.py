from hypoexponential import hypoexpon_gen
from scipy.stats import expon

import matplotlib.pyplot as plt
import numpy as np

# example
scales = [1.0, 0.5, 0.333, 0.25]
hypoexpon = hypoexpon_gen(scales)
xs = np.linspace(0, np.sum(scales)*10, 1000)

# plots of p.d.f.
for s in scales:
    plt.plot(xs, expon.pdf(xs, scale=s), label=f'Expon(1/λ={s})')
plt.plot(xs, hypoexpon.pdf(xs), label=f"Hypoexpon({', '.join(str(s) for s in scales)})")
plt.legend()
plt.suptitle('probability density functions')
plt.show()
