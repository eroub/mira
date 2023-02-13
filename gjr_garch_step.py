import argparse
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import helpers.analysis as analysis
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
nth_data = 30
num_std = 2
period = 500

# Init lists
merged_df_list = []
sectioned_data_list = []
bol_list = []
pred_bol_list = []
# Get the data
results = process_from_parquet_step(num_files, check_num, num_steps)
for i, (model_data, check_data) in enumerate(results):
    res = model_data.copy().iloc[1:].assign(
        diff_open=model_data['open'].diff(),
        diff_high=model_data['high'].diff(),
        diff_low=model_data['low'].diff(),
        diff_close=model_data['close'].diff(),
        datetime=pd.to_datetime(model_data['timestamp'], unit='ms')
    )

    # Specify and fit the GJR-GARCH model
    exogenous = res[['volume', 'datetime', 'diff_open', 'diff_high', 'diff_low']]
    gjr_garch_fit = arch_model(res['diff_close'], vol='GARCH', p=params[0], o=params[2], q=params[1], power=2, dist='t', mean='HAR', x=exogenous).fit(disp=False)

    # Generate predictions
    horizon = int((len(res) / num_files) / num_hours)
    forecast = gjr_garch_fit.forecast(horizon=horizon, simulations=1000)
    pred_vol, pred_mean = forecast.variance.iloc[-1], forecast.mean.iloc[-1]
    df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
    while True:
        pred_returns = t(df=df, loc=pred_mean, scale=np.sqrt(pred_vol)).rvs(size=horizon)
        pred_price = res['close'].iloc[-1] + pred_returns.cumsum()
        pred_df = pd.DataFrame({
            'pred_price': pred_price, 
            'timestamp': res['timestamp'].iloc[-1] + np.arange(1, len(pred_price) + 1) * (res['timestamp'].iloc[-1] - res['timestamp'].iloc[-2])
        })
        merged_df = pd.merge_asof(pred_df, check_data[['close', 'timestamp']], on='timestamp', direction='nearest', tolerance=15000).dropna().reset_index()
        if(len(merged_df) > 0): break
    # Convert the timestamps from milliseconds to datetime format
    merged_df['datetime'] = pd.to_datetime(merged_df['timestamp'], unit='ms')

    # Truncate the dataframe such that it only has timestamps within 'num_hours' hours from the first one
    time_diff = (merged_df.iloc[-1]['timestamp'] - merged_df.iloc[0]['timestamp']) / (3600 * 1000)
    if time_diff > num_hours:
        truncate_time = merged_df.iloc[0]['timestamp'] + num_hours * 3600 * 1000
        merged_df = merged_df.loc[merged_df['timestamp'] <= truncate_time]


    # Calculate the bollinger bands for the entire res DataFrame
    bol = analysis.bollinger_bands(res['close'], period, num_std)
    bol.index = res['datetime']

    # Concatenate the existing log_return data with the predicted data
    log_return = pd.Series(np.log(res['close'].diff()[1:]), index=res.index[1:])
    log_return.index = res.index[1:]
    log_return = log_return.reindex(pred_df.index)
    combined_data = pd.concat([pred_df], axis=0)
    combined_data['log_return'] = log_return

    # Calculate the cumulative sum of the log returns for the combined data
    cumulative_sum = np.exp(combined_data['log_return'].cumsum())

    # Scale the cumulative sum to the last known price
    scaled_cumulative_sum = cumulative_sum * res['close'].iloc[-1]

    # Calculate the upper and lower bands using the predicted mean and variance
    upper_band = scaled_cumulative_sum + num_std * np.sqrt(pred_vol)
    lower_band = scaled_cumulative_sum - num_std * np.sqrt(pred_vol)

    # Calculate the predicted moving average for the combined data
    pred_moving_avg = combined_data['pred_price'].rolling(window=period).mean()

    # Create dataframe for predicted bollinger bands
    pred_bol = pd.DataFrame({'upper_band': upper_band, 'moving_avg': pred_moving_avg, 'lower_band': lower_band})
    pred_bol = pred_bol.iloc[:len(merged_df), :]
    pred_bol.index = merged_df['datetime']

    print(combined_data['log_return'])
    print(pred_bol)

    # Select the last nth section of the historical dat and ema
    sectioned_data = res.iloc[-(res.shape[0]//nth_data):].copy()
    bol_sectioned = bol.iloc[-(res.shape[0]//nth_data):].copy()

    # Convert the timestamps from milliseconds to datetime format
    sectioned_data['datetime'] = pd.to_datetime(sectioned_data['timestamp'], unit='ms')

    # Save merged_df, sectioned_data, and ema 
    merged_df_list.append(merged_df)
    sectioned_data_list.append(sectioned_data)
    bol_list.append(bol_sectioned)
    pred_bol_list.append(pred_bol)

# Create a figure with num_steps subplots
cols = math.ceil(num_steps / 5)
fig, axs = plt.subplots(5, cols, figsize=(10, 15), sharex=True, sharey=True)
axs = axs.flatten()

# Iterate through the data and subplots and populate them
for i, (merged_df, sectioned_data, bol, pred_bol, ax) in enumerate(zip(merged_df_list, sectioned_data_list, bol_list, pred_bol_list, axs)):
    if i >= num_steps:
        ax.remove()
        continue
    
    # Add vertical lines at midnight for each unique date
    unique_dates = np.unique(merged_df['datetime'].dt.date)
    for date in unique_dates:
        midnight = pd.Timestamp(date).replace(hour=0, minute=0, second=0)
        ax.axvline(midnight, color='black', linestyle='dotted')
    
    # Plot the historical prices
    ax.plot(sectioned_data['datetime'], sectioned_data['close'], label='Historical Prices')
    # Plot the predicted prices
    ax.plot(merged_df['datetime'], merged_df['pred_price'], label='Predicted Prices')
    # Plot the actual prices
    ax.plot(merged_df['datetime'], merged_df['close'], label='Actual Prices', linestyle='dashed')
    # Plot the bollinger bands
    ax.plot(sectioned_data['datetime'], bol['upper_band'], linestyle='dotted', color='m')
    ax.plot(sectioned_data['datetime'], bol['lower_band'], linestyle='dotted', color='m')
    ax.plot(sectioned_data['datetime'], bol['moving_avg'], color='m')
    ax.fill_between(sectioned_data['datetime'], bol['upper_band'], bol['lower_band'], color='m', alpha=0.2)
    # Plot the predicted bollinger bands
    pred_bol_selected = pred_bol.iloc[:merged_df.shape[0]]
    # print(pred_bol_selected['upper_band'])
    # print(pred_bol_selected['lower_band'])
    # print(pred_bol_selected['moving_avg'])
    ax.plot(merged_df['datetime'], pred_bol_selected['upper_band'], linestyle='dotted', color='r')
    ax.plot(merged_df['datetime'], pred_bol_selected['lower_band'], linestyle='dotted', color='r')
    ax.plot(merged_df['datetime'], pred_bol_selected['moving_avg'], color='y')
    ax.fill_between(merged_df['datetime'], pred_bol_selected['upper_band'], pred_bol_selected['lower_band'], color='y', alpha=0.2)
    
    # Set labels
    ax.set_ylabel('Price')
    # ax.set_title(f'Predicted vs Actual Prices for Item {i+1}')
    
# Show date underneath last subplot
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y\n%H:%M:%S'))
# Add the legend to the figure
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper left', ncol=3)
# Adjust the subplot spacing
fig.tight_layout()
# Show the plot
plt.show()
