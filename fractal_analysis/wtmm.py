import glob
import random
import numpy as np
import pywt
import matplotlib.pyplot as plt
from preprocessing import grab_data

# Global Variables
frequency = 10000
num_files = 50

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

# Randomly select a start index
start_index = random.randint(0, len(file_names) - num_files)

# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]

# Get the data
res = grab_data(sample_file_names, frequency)

# perform the wavelet transform
wavelet = 'db3' # choose a wavelet function, 'db1' is the Daubechies wavelet of order 1
coeffs = pywt.wavedec(res['close'], wavelet)

# calculate the modulus maxima
cA, cD = coeffs[0], coeffs[1:]
wtmm = [np.max(np.abs(cd)) for cd in cD]

# define the scales
scales = range(1, len(wtmm) + 1)

# create the wtmm plot
plt.plot(scales, wtmm, 'o-')

plt.xlabel('Scale')
plt.ylabel('WTMM')
plt.title('WTMM plot')
plt.legend(['WTMM'])
plt.show()