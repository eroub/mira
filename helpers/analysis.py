import numpy as np

def directional_accuracy(y_true, y_pred):
    n = len(y_true)
    num_correct = 0
    for i in range(n-1):
        if (y_true[i + 1] - y_true[i]) * (y_pred[i + 1] - y_pred[i]) >= 0:
            num_correct += 1
    return num_correct / n

def directional_accuracy_bins(merged_df, num_bins):
    n = len(merged_df)
    num_correct = 0
    for i in range(0, n-num_bins, num_bins):
        y_true_mean = merged_df['close'][i:i+num_bins].mean()
        y_pred_mean = merged_df['pred_price'][i:i+num_bins].mean()
        if (y_true_mean - merged_df['close'][i]) * (y_pred_mean - merged_df['pred_price'][i]) >= 0:
            num_correct += 1
    return num_correct / (n / num_bins)

# Mostly useless since the price will never be negative
def directional_symmetry(y_true, y_pred):
    n = len(y_true)
    num_same = 0
    for i in range(n):
        if (y_true[i + 1] - y_true[i]) * (y_pred[i + 1] - y_pred[i]) >= 0:
            if abs(y_true[i]) == abs(y_pred[i]):
                num_same += 1
    return num_same / n

def rmse(merged_df):
    residual = (merged_df['pred_price'] - merged_df['close']) ** 2
    return np.sqrt(np.mean(residual))

def mse(merged_df):
    residual = (merged_df['pred_price'] - merged_df['close']) ** 2
    return np.mean(residual)

def mape(merged_df):
    residual = np.abs((merged_df['pred_price'] - merged_df['close']) / merged_df['close'])
    return np.mean(residual) * 100