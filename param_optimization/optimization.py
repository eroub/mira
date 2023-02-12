import sys, os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import math
import helpers.analysis as analysis
from collections import Counter
from tqdm import tqdm
from scipy.stats import t
from arch import arch_model
from arch.__future__ import reindexing
from helpers.preprocessing import process_from_parquet

# Global Variables
num_files = 21
check_num = 1
num_hours = 3
tries = 750
# Model Variables
# 
genetic_res_mean_median = [[2,0,1],[3,0,1],[1,0,3],[1,0,1],[3,0,0],[0,2,1],[3,0,3],[0,1,1],[2,0,0],[1,0,0]]
genetic_res_rmse = [[3,0,2],[0,3,3],[0,1,1],[3,0,1],[0,1,2],[0,2,1],[0,2,0],[0,3,0]]
genetic_res_da = [[0,3,2],[2,2,0],[0,1,3],[1,3,0],[1,0,0],[0,2,3],[0,3,1],[0,2,1],[0,2,0],[3,1,0]]

# Define the prediction algorithm
def pred_algo(params, choice):
    res, check_data = process_from_parquet(num_files, check_num)
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

    # Return directional accuracy or RSME based on user choice
    if choice == 0:
        return analysis.directional_accuracy(merged_df['close'], merged_df['pred_price'])
    else:
        return analysis.rmse(merged_df)

result_dict = {}
print("What sort of optimization?")
opt_type = int(input("0: Directional Accuracy - 1: Root Mean Square Error"))
if opt_type != 1 and opt_type != 0: exit()
for param in tqdm(genetic_res_da):
    i = 0
    arr = []
    while i < tries:
        residual = pred_algo(param, opt_type)
        # If residual is wack result try again
        if residual == 0 or residual is None or math.isnan(residual):
            continue
        # Otherwise add to array
        else:
            arr.append(residual)
            i += 1
    # Aferward average residual array and push results onto result_dict
    result_dict[str(param)] = sum(arr)/len(arr)

sorted_d = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_d)