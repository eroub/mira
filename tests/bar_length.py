import glob
import random
import pandas as pd
from tqdm import tqdm
from helpers.preprocessing import grab_data

# Global Variables
frequency = 500
num_files = 21
num_iterations = 50
mean_diff_list = []
median_diff_list = []
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

    # Preprocess and grab data, return in 'res'
    res = grab_data(sample_file_names, frequency)

    # Create a new column that contains the time difference between each entry and the previous entry
    # Also convert milliseconds to minutes
    res['time_diff'] = (res['timestamp'].diff())/60000

    # Calculate the mean, median, and mode of the time difference column
    mean_time_diff = res['time_diff'].mean()
    median_time_diff = res['time_diff'].median()
    mean_diff_list.append(mean_time_diff)
    median_diff_list.append(median_time_diff)

final_mean = sum(mean_diff_list) / num_iterations
final_median = sum(median_diff_list) / num_iterations

print("Final Mean: ", final_mean)
print("Final Median: ", final_median)