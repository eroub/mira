import glob
import random
import matplotlib.pyplot as plt
from preprocessing import grab_data
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# Global Variables
frequency = 500
num_files = 7

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))

# Randomly select a start index
start_index = random.randint(0, len(file_names) - num_files)

# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]

# Get the data
res = grab_data(sample_file_names, frequency)

# Calculate the difference between consecutive closing values
res['diff_close'] = res['close'].diff()
# Drop first value which is NaN
res = res.iloc[1:]

# Create a function to perform the ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# Create a function to perform the KPSS test
def kpss_test(series):
    result = kpss(series)
    print('KPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[3].items():
        print('\t%s: %.3f' % (key, value))

print("ADF on closing values")
print("---")
adf_test(res['close'])
print("---")
print("ADF on differenced closing values")
print("---")
adf_test(res['diff_close'])
print("---")

# print("KPSS on closing values")
# print("---")
# kpss_test(res['close'])
# print("---")
# print("KPSS on differenced closing values")
# print("---")
# kpss_test(res['diff_close'])
# print("---")