import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from preprocessing import grab_data

# Global Variables
frequency = 5000
num_files = 7

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

# Randomly select a start index
start_index = random.randint(0, len(file_names) - num_files)

# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]

res = grab_data(sample_file_names, frequency)
print (res)

# Compute the cumulative returns
res['returns'] = np.log(res['close']).cumsum()

# convert timestamp to datetime object
res['timestamp'] = pd.to_datetime(res['timestamp'], unit='ms')

# set timestamp as the index
res.set_index('timestamp', inplace=True)

# Define a linear function to be used for fitting
def linear(x,a,b):
    return a*x + b

# Set variable on what we want to plot
to_plot = res['close']

#Fit the linear function to the data
popt, _ = curve_fit(linear, range(len(to_plot)), to_plot)
# get the slope of the trendline
slope = popt[0]

# plot the log returns as a function of timestamp
plt.plot(res.index, to_plot, 'o', label='Original Data')
plt.xlabel('Timestamp')
plt.ylabel('Log Returns')

# plot the trendline
plt.plot(res.index, linear(range(len(to_plot)),*popt), '--',label='Trendline')

# show the slope of the trendline on the chart
plt.annotate("Slope: {:.3f}".format(popt[0]), xy=(0.8, 0.05), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# format the x-axis tick labels
xfmt = mdates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(xfmt)
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()
