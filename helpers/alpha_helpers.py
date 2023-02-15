import numpy as np
import pandas as pd

# Helper function to compute rolling statistics
def rolling(fn, x, d):
    return x.rolling(d).apply(fn)

# Rank of elements in each column of x
def rank(x):
    return x.rank(axis=0)

# Value of x d days ago
def delay(x, d):
    return x.shift(d)

# Time-serial correlation of x and y for the past d days
def correlation(x, y, d):
    return rolling(lambda a, b: np.corrcoef(a, b)[0,1], pd.concat([x, y], axis=1), d)

# Time-serial covariance of x and y for the past d days
def covariance(x, y, d):
    return rolling(lambda a, b: np.cov(a, b)[0,1], pd.concat([x, y], axis=1), d)

# Rescaled x such that sum(abs(x)) = a
def scale(x, a=1):
    return a * x / np.sum(np.abs(x))

# Today’s value of x minus the value of x d days ago
def delta(x, d):
    return x.diff(d)

# x raised to the power of a
def signedpower(x, a):
    return np.sign(x) * np.abs(x) ** a

# Weighted moving average over the past d days with linearly decaying weights d, d – 1, ..., 1
def decay_linear(x, d):
    weights = np.arange(1, d+1)[::-1]
    weights = weights / np.sum(weights)
    return rolling(lambda a: np.sum(a * weights), x, d)

# Cross-sectionally neutralized against groups g
def indneutralize(x, g):
    group_means = x.groupby(g).mean()
    return x.sub(group_means.loc[g].values, axis=1)

# Time-series statistics over the past d days
def ts_op(x, fn, d):
    return rolling(fn, x, d)

def ts_min(x, d):
    return ts_op(x, np.min, d)

def ts_max(x, d):
    return ts_op(x, np.max, d)

def ts_argmin(x, d):
    return ts_op(x, np.argmin, d)

def ts_argmax(x, d):
    return ts_op(x, np.argmax, d)

def ts_rank(x, d):
    return ts_op(x, lambda a: rank(a)[-1], d)

def min(x, d):
    return ts_min(x, d)

def max(x, d):
    return ts_max(x, d)

def sum(x, d):
    return ts_op(x, np.sum, d)

def product(x, d):
    return ts_op(x, np.prod, d)

def stddev(x, d):
    return ts_op(x, np.std, d)

# Volume-weighted average price
def vwap(data):
    return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

# Average daily dollar volume for the past d days
def adv(data, d):
    # Compute the rolling VWAP
    vwap = (data['close'] * data['volume']).rolling(d).sum() / data['volume'].rolling(d).sum()
    # Compute the average daily dollar volume
    return data['volume'].rolling(d).mean() * vwap
