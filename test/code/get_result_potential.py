import pandas as pd
import numpy as np
import time
import math
import sys
import os.path
from numba import njit, float64
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import src.statistics_observables as stat
from astropy.stats import jackknife_resampling, jackknife_stats, bootstrap
import itertools


def get_field(data, df1, df, time_size_max):

    time_size = data["T"].iloc[0]
    space_size = data["r/a"].iloc[0]

    # print("time_size", time_size)
    # print("space_size", space_size)

    if time_size < time_size_max:

        x1 = data['wilson_loop'].to_numpy()

        x2 = df[(df["T"] == time_size + 1) & (df["r/a"]
                                              == space_size)]['wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        # field, err = stat.jackknife_var(x3, potential)
        field, err = stat.jackknife_var_numba(x3, potential_numba)

        # print("my jackknife", stat.jackknife_var_numba(x1.reshape((1, x1.shape[0])), stat.average))

        # print("astropy jackknife", jackknife_stats(x1, np.mean, 0.95))

        # print("my bootstrap", stat.bootstrap_numba(x1.reshape((1, x1.shape[0])), stat.average, 100000))

        # print("astropy bootstrap", bootstrap(x1, 1000, bootfunc=np.mean))

        new_row = {'aV(r)': field, 'err': err}

        df1 = df1.append(new_row, ignore_index=True)

        return df1


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
    return y


def get_bin_borders(data_size, bin_size):
    nbins = data_size // bin_size
    bin_sizes = [bin_size for _ in range(nbins)]
    residual_size = data_size - nbins * bin_size
    idx = 0
    while residual_size > 0:
        bin_sizes[idx] += 1
        residual_size -= 1
        idx = (idx + 1) % nbins
    return np.array([0] + list(itertools.accumulate(bin_sizes)))


mu1 = ['0.05']
chains = {"s0", "s1", "s2", "s3", "s4"}

# for mu in mu1:
#     data = []
#     for chain in chains:
#         for i in range(0, 700):
#             file_path = f"../data/wilson_loop/qc2dstag/40^4/mu{mu}/{chain}/wilson_loop_{i:04}"

#             # print(file_path)
#             # print(chain)
#             if(os.path.isfile(file_path)):
#                 data.append(pd.read_csv(file_path, header = 0, names=["T", "r/a", "wilson_loop"]))
#                 data[-1]["conf_num"] = i

#     df = pd.concat(data)

#     # df = df[(df['T'] >= 10) & (df['T'] <= 11)]
#     # df = df[df['r/a'] == 10]

#     df1 = pd.DataFrame(columns=["aV(r)", "err"])

#     time_size_max = df["T"].max()

#     start = time.time()

#     df1 = df.groupby(['T', 'r/a']).apply(get_field, df1, df, time_size_max).reset_index()

#     end = time.time()
#     print("execution time = %s" % (end - start))

#     df1 = df1[['T', 'r/a', 'aV(r)', 'err']]

#     path_output = f"../result/potential/qc2dstag/40^4"

#     try:
#         os.makedirs(path_output)
#     except:
#         pass

#     df1.to_csv(f"{path_output}/potential1_mu={mu}.csv", index=False)

s_test = np.array([np.array([2.67, 3.1431, 5.234234, 3.22534,
                             1.135231, 8.4352345, 4.345235, 9.234234])])

print(stat.jackknife_var_numba_binning(
    s_test, trivial, get_bin_borders(len(s_test[0]), 3)))
