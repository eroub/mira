import random
import pandas as pd
import numpy as np
import math
from collections import Counter
from tqdm import tqdm
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet

# Global Variables
num_files = 21
check_num = 1
num_hours = 6
tries = 1000
# Model Variables
genetic_res = [[2,0,1],[3,0,1],[1,0,3],[1,0,1],[3,0,0],[0,2,1],[3,0,3],[0,1,1],[2,0,0],[1,0,0]]

# Define the prediction algorithm
def pred_algo(params):
    res, check_data = process_from_parquet(num_files, check_num)

    # Calculate the difference between consecutive closing values
    res['diff_close'] = res['close'].diff()
    # Drop first value which is NaN
    res = res.iloc[1:]

    # Convert timestamp to datetime format
    res['datetime'] = pd.to_datetime(res['timestamp'], unit='ms')
    # Extract minute of the day from datetime
    res['minute'] = res['datetime'].dt.minute

    # Specify the AMIRA-GARCH model
    amira_garch = arch_model(res['diff_close'], vol='GARCH', p=params[0], o=params[1], q=params[2], power=2, dist='t', mean='HAR', x=res[['volume', 'datetime']])
    # Fit the model
    amira_garch_fit = amira_garch.fit(disp=False)

    # Determine the length of half a day in terms of the time steps of your dataset
    horizon = int((len(res) / num_files) / 2)

    # Generate predictions for the next 'horizon' time steps
    forecast = amira_garch_fit.forecast(horizon=horizon, simulations=1000)
    predicted_values = forecast.simulations.values
    pred_mean = forecast.mean.iloc[-1]
    pred_vol = forecast.variance.iloc[-1]
    # Pass the estimated df to the t distribution
    df = (len(res['close'])-1) * (np.var(res['close']) + (np.mean(res['close'])-0)**2)/(np.var(res['close'])/(len(res['close'])-1) + (np.mean(res['close'])-0)**2)
    dist = t(df=df, loc=pred_mean, scale=np.sqrt(pred_vol))
    pred_returns = dist.rvs(size=horizon)
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

    # Truncate the dataframe such that it only has timestamps within 6 hours from the first one
    if time_diff > num_hours:
      truncate_time = start_time + num_hours * 3600 * 1000
      merged_df = merged_df.loc[merged_df['timestamp'] <= truncate_time]

    # Group truncated merged dataframe into bins according to the num_hours
    bins = pd.cut(merged_df['timestamp'], num_hours)
    grouped_df = merged_df.groupby(bins)

    # Calculate the mean and median of the residual for each bin
    merged_df['residual'] = ((merged_df['pred_price'] - merged_df['close']) / merged_df['close'] ) * 100
    grouped_df = merged_df.groupby(bins)
    agg_stats = grouped_df['residual'].agg(['mean', 'median'])

    residual_average = np.mean(np.abs(agg_stats['median'].values))

    return residual_average

result_dict = {}
for param in tqdm(genetic_res):
    i = 0
    arr = []
    while i < tries:
        residual = pred_algo(param)
        # If residual is wack result try again
        if residual == 0 or residual is None or math.isnan(residual):
            continue
        # Otherwise add to array
        else:
            arr.append(residual)
            i += 1
    # Aferward average residual array and push results onto result_dict
    result_dict[str(param)] = sum(arr)/len(arr)

sorted_d = sorted(result_dict.items(), key=lambda x: x[1], reverse=False)
print(sorted_d)