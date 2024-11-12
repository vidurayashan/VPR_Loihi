import numpy as np
from scipy.stats import norm

def randn2(r, c):
    mat = []
    for i in range(r):
        # mat.append(np.round(norm.ppf(np.random.rand(c)), 4))
        mat.append(norm.ppf(np.random.rand(c)))
    return np.array(mat)