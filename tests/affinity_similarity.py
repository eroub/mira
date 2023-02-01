import glob
import random
import numpy as np
import pandas as pd
from helpers.preprocessing import grab_data
from numpy.polynomial import polynomial as poly
from scipy.stats import linregress
from tqdm import tqdm

# Global Variables
frequency = 2500
num_files = 40
num_iterations = 100
scaling_list = []
used_indices = set()

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

for i in tqdm(range(num_iterations)):

    # Randomly select a start index
    start_index = random.randint(0, len(file_names) - num_files)
    # Check if the start index is already in the set of used indices
    while start_index in used_indices:
        start_index = random.randint(0, len(file_names) - num_files)
    # Add the start index to the set of used indices
    used_indices.add(start_index)

    # Get a contiguous subset of [num_files] file names
    sample_file_names = file_names[start_index : start_index + num_files]

    res = grab_data(sample_file_names, frequency)

    # Compute the cumulative returns
    res['returns'] = np.log(res['close']).cumsum()

    # Divide the time series into segments of a specified length
    window_size = 128
    segments = [res.iloc[i:i+window_size]['close'] for i in range(0, len(res)-window_size, window_size)]

    # Detrend each segment by fitting and subtracting a polynomial
    degree = 1
    for i in range(len(segments)):
        segments[i] = segments[i] - poly.polyfit(range(window_size), segments[i], degree)[0]

    # Compute the root mean square fluctuation for each segment
    rmss = [np.sqrt(np.mean(segment**2)) for segment in segments]

    # Compute the scaling exponent by fitting a line to the log-log plot
    slope, _, _, _, _ = linregress(np.log(range(1, len(segments)+1)), np.log(rmss))
    scaling_exponent = slope

    scaling_list.append(scaling_exponent)

# Compute average of scaling_exponents
scaling_avg = sum(scaling_list) / num_iterations

# Compare the scaling exponent to a threshold value
threshold = 0.5
print(scaling_avg)
if scaling_avg > threshold:
    print("The data exhibits self-affinity and self-similarity.")
else:
    print("The data does not exhibit self-affinity and self-similarity.")
