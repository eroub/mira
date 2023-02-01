import glob
import random
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global Variables
frequency = 1000
num_files = 21

def convert_to_tick(data):
    # Group the dataframe by every 1000 rows
    df_grouped = data.groupby(data.index // frequency)

    # Compute the open, high, low, close, volume, and timestamp for each group
    df_agg = df_grouped.agg({'price': ['first', 'max', 'min', 'last'],
                            'volume': 'sum',
                            'timestamp': 'last'})
    # Rename the columns
    df_agg.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    # Reset the index
    df_agg.reset_index(drop=True, inplace=True)
    # Return df_agg
    return df_agg



# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

# Randomly select a start index
start_index = random.randint(0, len(file_names) - num_files)

# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]

# Initialize an empty dataframe
subset_data = pd.DataFrame()

# Loop through the selected file names
for file_name in tqdm(sample_file_names):
    # Read in the data from the file
    data = pd.read_csv(file_name, usecols=[1,2,4], names=["price", "volume", "timestamp"])
    # Call convert_to_tick for each file
    res = convert_to_tick(data)
    # Append the data to the array
    subset_data = pd.concat([subset_data, res], axis=0)

# Compute the cumulative returns
res['returns'] = np.log(res['close']).cumsum()

# Perform wavelet decomposition
coeffs = pywt.wavedec(res['returns'], 'db4')

# Reconstruct the original signal
reconstructed_signal = pywt.waverec(coeffs, 'db4')

# Apply thresholding
thresholded_coeffs = pywt.threshold(coeffs, mode='soft')

# Perform continuous wavelet transform
cwtmatr, freqs = pywt.cwt(res['returns'], 2, 'mexh')

# Plot the coefficients
for i, coeff in enumerate(coeffs):
    plt.stem(coeff, linefmt='C{}-'.format(i), markerfmt='C{}o'.format(i))
plt.show()

# Plot the reconstructed signal
plt.plot(reconstructed_signal)
plt.show()

plt.plot(res['returns'],label='Original signal')
plt.plot(reconstructed_signal,label='Reconstructed signal')
plt.legend()
plt.show()

plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.colorbar()
plt.show()