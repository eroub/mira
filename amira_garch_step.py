import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import helpers.analysis as analysis
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet_step

# Global Variables
num_files = 21
check_num = 1
num_hours = 2
num_steps = 5
# Model Variables
# According to my testing the best parameters for BTC 200 tick data is [0,2,1,2]
p_symmetric_lag = [0,1,2,3]
o_asymmetric_lag = [0,1,2,3]
q_volatility_lag = [0,1,2,3]

# Get the data
merged_df_list = []
sectioned_data_list = []
results = process_from_parquet_step(num_files, check_num, num_steps)
for i, (model_data, check_data) in enumerate(results):
    locals()[f"model_data_{i+1}"] = model_data
    locals()[f"check_data_{i+1}"] = check_data

for i in range(1, num_steps+1):
    model_data = eval(f"model_data_{i}")
    check_data = eval(f"check_data_{i}")
    # Calculate the difference between consecutive OHLC values
    res = model_data.copy()
    res['diff_open'] = res['open'].diff()
    res['diff_high'] = res['high'].diff()
    res['diff_low'] = res['low'].diff()
    res['diff_close'] = res['close'].diff()
    # Drop first value which is NaN and convert timestamp to datetime format
    res = res.iloc[1:]
    res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')

    # Specify the AMIRA-GARCH model and fit it
    exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
    amira_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=p_symmetric_lag[0], o=o_asymmetric_lag[2], q=q_volatility_lag[1], power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)

    # Determine the length of half a day in terms of the time steps of your dataset
    horizon = int((len(res) / num_files) / num_hours)

    # Generate predictions for the next 'horizon' time steps
    forecast = amira_garch_fit.forecast(horizon=horizon, simulations=1000)
    pred_mean = forecast.mean.iloc[-1]
    pred_vol = forecast.variance.iloc[-1]

    print(f"---{i} MEAN---")
    print(pred_mean)
    print(f"---{i} VOL---")
    print(pred_vol)

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

    # Select the last section of the historical data
    sectioned_data = res.iloc[-(res.shape[0]//30):]

    # Save merged_df and sectioned_data
    merged_df_list.append(merged_df)
    sectioned_data_list.append(sectioned_data)

# Create a figure with three subplots
fig, axs = plt.subplots(num_steps, figsize=(10, 15), sharex=True, sharey=True)

for i, (merged_df, sectioned_data, ax) in enumerate(zip(merged_df_list, sectioned_data_list, axs)):
    # Plot the historical prices
    ax.plot(sectioned_data['timestamp'], sectioned_data['close'], label='Historical Prices')
    # Plot the predicted prices
    ax.plot(merged_df['timestamp'], merged_df['pred_price'], label='Predicted Prices')
    # Plot the actual prices
    ax.plot(merged_df['timestamp'], merged_df['close'], label='Actual Prices', linestyle='dashed')
    # Add labels and legends
    # ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    # Set the title of each subplot
    # ax.set_title(f'Predicted vs Actual Prices for Item {i+1}')

# Adjust the subplot spacing
fig.tight_layout()
# Show the plot
plt.show()
