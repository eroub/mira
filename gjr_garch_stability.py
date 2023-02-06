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
print(amira_garch_fit.summary())

# Determine the length of half a day in terms of the time steps of your dataset
horizon = int((len(res) / num_files) / num_hours)

# Generate predictions for the next 'horizon' time steps
forecast = amira_garch_fit.forecast(horizon=horizon, simulations=1000)
pred_mean = forecast.mean.iloc[-1]
pred_vol = forecast.variance.iloc[-1]

# Compute the standard deviation for the predictions
analysis.standard_deviation(horizon, pred_vol, pred_mean, 0.05, amira_garch_fit.params['nu'], t)

print(pred_mean)
print(pred_vol)