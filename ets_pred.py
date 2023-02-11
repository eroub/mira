import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet
from scipy.stats import t
import statsmodels.api as sm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=10, help="number of steps")
parser.add_argument("--num-files", type=int, default=21, help="number of input data files")
parser.add_argument("--check-num", type=int, default=21, help="number of check data files")
parser.add_argument("--num-hours", type=int, default=6, help="number of hours to check predicted vs actual data against")
args = parser.parse_args()
# Global Variables
num_steps = args.num_steps
num_files = args.num_files
check_num = args.check_num
num_hours = args.num_hours
num_price_gens = 1

# Model Variables
# According to my testing the best parameters for BTC 200 tick data is [0,2,1,2]
p_symmetric_lag = [0,1,2,3]
o_asymmetric_lag = [0,1,2,3]
q_volatility_lag = [0,1,2,3]

# Get the data
merged_df_list = []
sectioned_data_list = []
res, check_data = process_from_parquet(num_files, check_num)

res['diff_open'] = res['open'].diff()
res['diff_high'] = res['high'].diff()
res['diff_low'] = res['low'].diff()
res['diff_close'] = res['close'].diff()
# Drop first value which is NaN and convert timestamp to datetime format
res = res.iloc[1:]
res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')

# Specify the GJR-GARCH model and fit it
exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
gjr_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=p_symmetric_lag[0], o=o_asymmetric_lag[2], q=q_volatility_lag[1], power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)

# Determine the length of half a day in terms of the time steps of your dataset
horizon = int((len(res) / num_files) / num_hours)
# horizon = 10000

# Generate predictions for the next 'horizon' time steps
forecast = gjr_garch_fit.forecast(horizon=horizon, simulations=1000)
pred_mean = forecast.mean.iloc[-1]
pred_vol = forecast.variance.iloc[-1]

# Fit the Exponential Smoothing State Space Model (ETS)
res['close'].index = pd.to_datetime(res['close'].index)
ets = sm.tsa.ExponentialSmoothing(res['close'], trend='add', seasonal='multiplicative', seasonal_periods=int(len(res['close'])/num_hours)).fit()
pred_price = ets.forecast(horizon)

print(pred_price)
print("LEN pred_prices: {}".format(len(pred_price)))
print("LEN:gjr preds: {}".format(len(pred_mean)))

# Combine the predicted mean and volatility series from the GJR-GARCH model with the ETS model forecast
simulated_prices = pred_price.values * np.sqrt(pred_vol.values) + pred_mean.values

# Create a new x-axis that starts where the historical x-axis ends
x_pred = res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])

# Create data frame from the pred_price and x_pred
data = {'pred_price': pred_price, 'timestamp': x_pred}
pred_df = pd.DataFrame(data)
merged_df = pd.merge_asof(pred_df, check_data[['close', 'timestamp']], on='timestamp', direction='nearest', tolerance=15000).dropna().reset_index()
start_time = merged_df.iloc[0]['timestamp']
end_time = merged_df.iloc[-1]['timestamp']
time_diff = (end_time - start_time) / (3600 * 1000)


# Truncate the dataframe such that it only has timestamps within 'num_hours' hours from the first one
if time_diff > num_hours:
    truncate_time = start_time + num_hours * 3600 * 1000
    merged_df = merged_df.loc[merged_df['timestamp'] <= truncate_time]


# Select the last section of the historical data
sectioned_data = res.iloc[-(res.shape[0]//30):]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))
# Plot the historical prices
ax.plot(sectioned_data['timestamp'], sectioned_data['close'], label='Historical Prices')
# Plot the predicted prices
ax.plot(merged_df['timestamp'], merged_df['pred_price'], label='Predicted Prices')
# Plot the actual prices
ax.plot(merged_df['timestamp'], merged_df['close'], label='Actual Prices', linestyle='dashed')
# Add labels and legends
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
# Set the title of the figure
fig.suptitle('Predicted vs Actual Prices')
# Show the plot
plt.show()