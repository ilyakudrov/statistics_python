import math
import numpy as np
from numba import njit, float64

# @njit


def mysum(xs):
    tmp = 0
    for x in xs:
        tmp += x

    return tmp

# @njit


def jackknife(x, func):
    n = x.shape[1]
    idx = np.arange(n)
    return mysum([func(x[:, idx != i]) for i in range(n)]) / float(n)

# @njit


def jackknife_var(x, func):
    n = x.shape[1]
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n - 1) / (n + 0.0) * mysum([(func(x[:, idx != i]) - j_est)**2.0
                                               for i in range(n)])


@njit
def bootstrap_smaple_generate(x, func, K):
    n = x.shape[1]
    m = x.shape[0]
    y = np.zeros((m, K))

    for k in range(m):
        for i in range(K):
            for j in range(n):
                y[k][i] += x[k][np.random.randint(0, n)]

    for k in range(m):
        for i in range(K):
            y[k][i] = y[k][i] / n

    return func(y)


@njit
def bootstrap_numba(x, func, K):
    sample = bootstrap_smaple_generate(x, func, K)
    estimate = sample.mean()
    sigma = 0
    n = sample.shape[0]
    for i in range(n):
        sigma += (sample[i] - estimate)**2
    return estimate, math.sqrt((n - 1) / (n + .0) * sigma)


@njit
def jackknife_sample_generate(x, func):
    m = x.shape[0]
    n = x.shape[1]
    sum_x = np.zeros(m)
    y = np.zeros((m, n))
    for j in range(m):
        for i in range(n):
            sum_x[j] += x[j][i]

    for j in range(m):
        for i in range(n):
            y[j][i] = (sum_x[j] - x[j][i]) / (n - 1)

    return func(y)


@njit
def jackknife_var_numba(x, func):
    n = x.shape[1]
    x1 = jackknife_sample_generate(x, func)
    j_est = x1.mean()
    sigma = 0
    for i in range(n):
        sigma += (x1[i] - j_est)**2
    return j_est, math.sqrt((n - 1) / (n + .0) * sigma)


@njit
def average(x):
    return x[0]
