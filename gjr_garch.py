import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyflux as pf
from helpers.preprocessing import grab_data
from scipy.stats import t

# Global Variables
frequency = 100
num_files = 21
check_num = 3

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))
# Randomly select a start index
# We want the last few days for checking against the forecasted values
start_index = random.randint(0, len(file_names) - num_files - check_num + 1)
# Get a contiguous subset of [num_files] file names
sample_file_names = file_names[start_index : start_index + num_files]
check_file_names = file_names[start_index+num_files : start_index+num_files+check_num-1]
# Get the data
res = grab_data(sample_file_names, frequency)
check_data = grab_data(check_file_names, frequency)

# Calculate the difference between consecutive closing values
res['diff_close'] = res['close'].diff()
# Drop first value which is NaN
res = res.iloc[1:]

# Convert timestamp to datetime format
res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')
# Extract hour of the day from datetime
res['hour'] = res['datetime'].dt.hour

# Define the p, d, q (pdq) and P, D, Q (PDQs) hyperparameters for the ARIMA model
pdq = (1, 0, 0)
PDQs = (1, 0, 0, 1)

# Use auto_arima to determine the best p, d, q and P, D, Q values
stepwise_fit = pm.auto_arima(res['diff_close'], start_p=pdq[0], start_q=pdq[1], max_p=pdq[0], max_q=pdq[1],
                          start_P=PDQs[0], start_Q=PDQs[1], max_P=PDQs[0], max_Q=PDQs[1], seasonal=True,
                          stepwise=True, suppress_warnings=True)

# Print the best p, d, q and P, D, Q values
print(stepwise_fit.order)

# Define the GJR-GARCH model
gjr_garch = pm.model_selection.GARCH(res['diff_close'], p=1, o=0, q=1, dist='t', 
                                      exog = res[['volume', 'hour']])

# Fit the model
gjr_garch_fit = gjr_garch.fit()

# Determine the length of a day in terms of the time steps of your dataset
horizon = int(len(res) / 21)

# Generate predictions for the next 'horizon' time steps
forecast = gjr_garch_fit.forecast(horizon=horizon, simulations=1000)
predicted_values = forecast.simulations.values
pred_mean = forecast.mean.iloc[-1]
pred_vol = forecast.variance.iloc[-1]
pred_returns = np.random.normal(loc=pred_mean, scale=np.sqrt(pred_vol), size=horizon)
pred_price = res['close'].iloc[-1] + pred_returns.cumsum()

# Compute the standard deviation for the predictions
std = pred_vol * np.sqrt(horizon)
# Calculate critical value
k = gjr_garch_fit.params['nu']
alpha = 0.05
critical_value = t.ppf(1 - alpha / 2, k)
print(critical_value)
# Compute the lower and upper bounds of the confidence interval
lower_bound = (pred_mean - critical_value * std) / 1000
upper_bound = (pred_mean + critical_value * std) / 1000
# Print the lower and upper bounds of the confidence interval
print("Lower bound of the confidence interval: {:.2f}%".format(lower_bound.iloc[-1]))
print("Upper bound of the confidence interval: {:.2f}%".format(upper_bound.iloc[-1]))

# Create a new x-axis that starts where the historical x-axis ends
x_pred = res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])
# Select the last third of the historical data
last_third_data = res.iloc[-(res.shape[0]//3):]
# Create a figure and a set of subplots
fig, axs = plt.subplots(2,1, figsize=(10, 10), sharex=True)
# Plot the historical prices on the both subplots
axs[0].plot(last_third_data['timestamp'], last_third_data['close'], label='Historical Prices')
# Plot the predicted prices on the left subplot
axs[0].plot(x_pred, pred_price, label='Predicted Prices')
axs[0].set_yticklabels([]) # remove ytick labels
axs[0].set_ylabel('Price')
axs[0].legend()
# Plot the historical prices on the right subplot
axs[1].plot(last_third_data['timestamp'], last_third_data['close'], label='Historical Prices')
# Plot the actual prices on the right subplot
axs[1].plot(check_data['timestamp'], check_data['close'], label='Actual Prices')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Price')
axs[1].legend()
# Overlay the two subplots in a single figure
fig.suptitle('Predicted vs Actual Prices')
plt.show()