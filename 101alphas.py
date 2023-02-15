import argparse
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import helpers.analysis as analysis
import alphas as a
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet_step
from scipy.stats import t

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=10, help="number of steps")
parser.add_argument("--num-files", type=int, default=21, help="number of input data files")
parser.add_argument("--check-num", type=int, default=2, help="number of check data files")
parser.add_argument("--num-hours", type=int, default=6, help="number of hours to check predicted vs actual data against")
args = parser.parse_args()
# Global Variables
num_steps = args.num_steps
num_files = args.num_files
check_num = args.check_num
num_hours = args.num_hours

# Model Variables
# According to my testing the best parameters for BTC 200 tick data is [0,2,1,2]
params = [0,1,2,3]
nth_data = 10
num_std = 2
period = 500
decay_rate = 0.1

# Init lists
check_data_list = []
sectioned_data_list = []
signal_list = []
# Get the data
results = process_from_parquet_step(num_files, check_num, num_steps)
signal_data = process_from_parquet_step(num_files, check_num, num_steps, 1000)
for i, (model_data, check_data) in enumerate(results):
    res, check_data = model_data.copy().iloc[1:], check_data.copy().iloc[1:]
    signal_res = signal_data[i][0]

    print(signal_res)
    print(res)

    signal = a.signal_101(signal_res)

    print(signal)

    # Select the last nth section of the historical data and signal
    sectioned_data = res.iloc[-(res.shape[0]//nth_data):].copy()
    sectioned_signal = signal.iloc[-(signal.shape[0]//nth_data):].copy()

    # Save check_data, sectioned_data, and sectioned_signal 
    check_data_list.append(check_data)
    sectioned_data_list.append(sectioned_data)
    signal_list.append(sectioned_signal)

# Create a figure with num_steps subplots
cols = math.ceil(num_steps / 5)
fig, axs = plt.subplots(5, cols, figsize=(10, 15), sharex=True, sharey=True)
axs = axs.flatten()

# Iterate through the data and subplots and populate them
for i, (check_data, sectioned_data, signal, ax) in enumerate(zip(check_data_list, sectioned_data_list, signal_list, axs)):
    if i >= num_steps:
        ax.remove()
        continue

    # Convert the timestamps from milliseconds to datetime format
    sectioned_data['datetime'] = pd.to_datetime(sectioned_data['timestamp'], unit='ms')
    check_data['datetime'] = pd.to_datetime(check_data['timestamp'], unit='ms')
    
    # Add vertical lines at midnight for each unique date
    unique_dates = np.unique(check_data['datetime'].dt.date)
    for date in unique_dates:
        midnight = pd.Timestamp(date).replace(hour=0, minute=0, second=0)
        ax.axvline(midnight, color='black', linestyle='dotted')
    
    # Plot the historical prices
    ax.plot(sectioned_data['datetime'], sectioned_data['close'], label='Historical Prices')
    # Plot the actual prices
    ax.plot(check_data['datetime'], check_data['close'], label='Actual future prices', c='r')

    # Create a new axis below the existing one
    ax2 = ax.twinx()
    # Plot the signal values on the new axis
    ax2.plot(sectioned_data['datetime'], signal, label='Signal', c='g')
    # Set the y-axis label for the new axis
    ax2.set_ylabel('Signal')
    # Set the limits of the new y-axis to be [-1, 1]
    ax2.set_ylim([-1, 1])
    # Shift the signal plot down to make it separate from the price plot
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0 - 0.15, ax2.get_position().width, ax2.get_position().height])

    
# Show date underneath last subplot
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m-%y'))
# Add the legend to the figure
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper left', ncol=3)
# Adjust the subplot spacing
fig.tight_layout()
# Show the plot
plt.show()
