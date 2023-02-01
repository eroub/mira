import glob
import random
import numpy as np
import nolds
from scipy.stats import linregress
from pyeeg import hurst
import pywt
from helpers.preprocessing import grab_data

# Global Variables
frequency = 2500
num_files = 7

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

# Randomly select a start index
start_index = random.randint(0, len(file_names) - num_files)

# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]

# Get the data
res = grab_data(sample_file_names, frequency)

# Compute the cumulative returns
res['returns'] = np.log(res['close']).cumsum()

def R_S(x):
    """Calculate the R/S statistic for a time series x."""
    n = len(x)
    x = np.cumsum(x - np.mean(x))
    R = np.max(x) - np.min(x)
    S = np.std(x)
    return R / S


def hurst_exponent(x):
    """Calculate the Hurst exponent for a time series x using R/S analysis."""
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
    return poly[0]*2.0

def DFA2(x, w=None, overlap=False):
    """Calculate the Hurst exponent for a time series x using DFA2."""
    if w is None:
        w = int(len(x) / 10)
    if overlap:
        w = int(w / 2)
    N = len(x)
    n = int(N / w)
    X = np.zeros((n, w))
    Y = np.zeros((n, w))
    for i in range(n):
        X[i, :] = np.arange(i * w, (i + 1) * w)
        Y[i, :] = x[i * w:(i + 1) * w]
    C = np.zeros(n)
    for i in range(n):
        p = np.polyfit(X[i, :], Y[i, :], 0)
        C[i] = p[0] * X[i, :] + p[1]
    F = np.zeros(n)
    for i in range(n):
        F[i] = np.sqrt(np.mean((Y[i, :] - C[i]) ** 2))
    n = np.arange(1, n + 1)
    poly = np.polyfit(np.log2(n), np.log2(F), 1)
    return poly[0]

def WTMM(x):
    """Calculate the Hurst exponent for a time series x using WTMM."""
    maxima = []
    for i in range(1, len(x)):
        cA, cD = pywt.dwt(x[:i], 'db1')
        maxima.append(np.max(np.abs(cD)))
    poly = np.polyfit(np.log10(range(1, len(x))), np.log10(maxima), 1)
    return poly[0]

def MFDFA(x, q=[2,3,4,5]):
    """Calculate the Hurst exponent for a time series x using MFDFA."""
    N = len(x)
    n = np.arange(1, N + 1, dtype=float)
    n = np.power(n, 2.0/3.0)
    F = np.zeros(N)
    for i in range(N):
        xi = x[i]
        F[i] = np.mean(np.abs(x[i:] - xi))
    poly = np.polyfit(np.log10(n), np.log10(F), 1)
    return poly[0]

def scipy_linregress(x): 
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
    slope, _, _, _, _ = linregress(np.log10(lags), np.log10(tau))
    return slope*2

x = res['close']
print("Range Scaled Analysis Hurst: ", hurst_exponent(x.tolist()))
print("PyEEG Hurst: ", hurst(x.tolist()))
# print("DFA2 Hurst: ", DFA2(x.tolist()))
print("Nolds DFA: ", nolds.dfa(x.tolist()))
# print("WTMM Hurst: ", WTMM(x.tolist()))
# print("MFDFA Hurst: ", MFDFA(x.tolist()))
print("Scipy Linregress: ", scipy_linregress(x.tolist()))