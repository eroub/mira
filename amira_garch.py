import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import helpers.analysis as analysis
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet

# Global Variables
num_files = 21
check_num = 1
num_hours = 3
# Model Variables
# According to my testing the best parameters for BTC 200 tick data is [0,2,1,2]
p_symmetric_lag = [0,1,2,3]
o_asymmetric_lag = [0,1,2,3]
q_volatility_lag = [0,1,2,3]

# Get the data
res, check_data = process_from_parquet(num_files, check_num)

# Calculate the difference between consecutive OHLC values
res['diff_open'] = res['open'].diff()
res['diff_high'] = res['high'].diff()
res['diff_low'] = res['low'].diff()
res['diff_close'] = res['close'].diff()
# Drop first value which is NaN# Drop first value which is NaN
res = res.iloc[1:]

# Convert timestamp to datetime format and then extract the minutes
res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')

# Specify the AMIRA-GARCH model and fit it
exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
# exogenous = res[['volume', 'datetime']]
amira_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=p_symmetric_lag[0], o=o_asymmetric_lag[2], q=q_volatility_lag[1], power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)
# Print the model summary
# print(amira_garch_fit.summary())

# Determine the length of half a day in terms of the time steps of your dataset
horizon = int((len(res) / num_files) / num_hours)

# Generate predictions for the next 'horizon' time steps
forecast = amira_garch_fit.forecast(horizon=horizon, simulations=1000)
pred_mean = forecast.mean.iloc[-1]
pred_vol = forecast.variance.iloc[-1]
# Pass the estimated degrees of freedom to the t distribution
df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
pred_returns = t(df=df, loc=pred_mean, scale=np.sqrt(pred_vol)).rvs(size=horizon)
pred_price = res['close'].iloc[-1] + pred_returns.cumsum()

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

# Group truncated merged dataframe into bins according to the num_hours
bins = pd.cut(merged_df['timestamp'], num_hours)
grouped_df = merged_df.groupby(bins)

# Calculate the RMSE of the residual for each bin
merged_df['residual'] = (merged_df['pred_price'] - merged_df['close']) ** 2
grouped_df = merged_df.groupby(bins)
agg_stats = grouped_df['residual'].agg(['mean', 'median'])
agg_stats['rmse'] = np.sqrt(agg_stats['mean'])
print("Root Mean Squared Error: ", agg_stats['rmse'].values)

# Also calculate the directional accuracy
# print(analysis.directional_accuracy(merged_df['close'], merged_df['pred_price']))
print(len(merged_df))
print(analysis.directional_accuracy_bins(merged_df, math.floor(math.sqrt(len(merged_df)))))

# Compute the standard deviation for the predictions
analysis.standard_deviation(horizon, pred_vol, pred_mean, 0.05, amira_garch_fit.params['nu'], t)

# Select the last tenth of the historical data
last_twentieth_data = res.iloc[-(res.shape[0]//20):]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))
# Plot the historical prices
ax.plot(last_twentieth_data['timestamp'], last_twentieth_data['close'], label='Historical Prices')
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

