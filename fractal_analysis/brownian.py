import glob
import random
import numpy as np
from pyeeg import hurst
import matplotlib.pyplot as plt
import scipy.signal
from preprocessing import grab_data

# Global Variables
frequency = 2500
num_files = 21

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

# calculate the Hurst exponent
H = hurst(res['close'])

# Brownian Motion
n = len(res['close'])
x = np.cumsum(res['close'])*np.power(n,-H/2)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1)

# Plot the line plot
axs[0].plot(x)
axs[0].set_title('Line plot')

# Plot the histogram
axs[1].hist(x, bins=50)
axs[1].set_title('Histogram')

# Plot the power spectral density
f, Pxx_den = scipy.signal.periodogram(x)
axs[2].semilogy(f, Pxx_den)
axs[2].set_title('Power Spectral Density')

# Show the figure
plt.show()