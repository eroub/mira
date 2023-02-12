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
nth_data = 20
# Model Variables
# According to my testing the best parameters for BTC 200 tick data is [0,2,1,2]
params = [0,1,2,3]

# Get the data
res, check_data = process_from_parquet(num_files, check_num)

# Calculate differences between consecutive OHLC values
res = res.assign(diff_open=lambda x: x['open'].diff(),
                 diff_high=lambda x: x['high'].diff(),
                 diff_low=lambda x: x['low'].diff(),
                 diff_close=lambda x: x['close'].diff(),
                 datetime=lambda x: pd.to_datetime(x['timestamp'], unit='ms')
                )[1:]

# Specify the GJR-GARCH model and fit it
exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
# exogenous = res[['volume', 'datetime']]
gjr_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=params[0], o=params[2], q=params[1], power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)
# Print the model summary
# print(gjr_garch_fit.summary())

# Determine the length of half a day in terms of the time steps of your dataset
horizon = int((len(res) / num_files) / num_hours)

# Generate predictions for the next 'horizon' time steps
forecast = gjr_garch_fit.forecast(horizon=horizon, simulations=1000)
# Calculate degrees of freedom for t distribution
df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
# Until merged_df returns data successfully generate price predictions
while True:
    # Generate price predictions using t distribution
    pred_returns = t(df=df, loc=forecast.mean.iloc[-1], scale=np.sqrt(forecast.variance.iloc[-1])).rvs(size=horizon)
    pred_price = res['close'].iloc[-1] + pred_returns.cumsum()

    # Create a new x-axis that starts where the historical x-axis ends
    x_pred = res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])

    # Create data frame from the pred_price and x_pred
    pred_df = pd.DataFrame({'pred_price': pred_price, 'timestamp': x_pred})
    merged_df = pd.merge_asof(pred_df, check_data[['close', 'timestamp']], on='timestamp', direction='nearest', tolerance=15000).dropna().reset_index()
    if(len(merged_df) > 0): break

# Truncate the dataframe such that it only has timestamps within 'num_hours' hours from the first one
time_diff = (merged_df.iloc[-1]['timestamp'] - merged_df.iloc[0]['timestamp']) / (3600 * 1000)
if time_diff > num_hours:
    truncate_time = merged_df.iloc[0]['timestamp'] + num_hours * 3600 * 1000
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
analysis.standard_deviation(horizon, forecast.variance.iloc[-1], forecast.mean.iloc[-1], 0.05, gjr_garch_fit.params['nu'], t)

# Select the nth section of the historical data
sectioned_data = res.iloc[-(res.shape[0]//nth_data):]

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

