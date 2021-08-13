import math
import numpy as np

def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n))/float(n)


def jackknife_var(x, func):
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n-1)/(n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
                                        for i in range(n))

def average(x):
    a = x.mean(axis=0)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0