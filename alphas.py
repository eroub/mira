import math
import helpers.alpha_helpers as h
import pandas as pd
import numpy as np

def signal_41(df):
    return math.pow((df['high'].iloc[-1]*df['low'].iloc[-1]), 0.5) - h.vwap(df).iloc[-1]

def signal_101(df):
    return ((df['close']-df['open'])/((df['high']-df['low'])+0.001))

def signal_49(df):
    close = df['close']
    delayed_close_1 = h.delay(close, 1)
    delayed_close_10 = h.delay(close, 10)
    delayed_close_20 = h.delay(close, 20)
    signal = np.where(
        (((((delayed_close_20 - delayed_close_10) / 10) - ((delayed_close_10 - close) / 10)) < (-1 * 0.1)) == True),
        1,
        (-1 * 1) * (close - delayed_close_1)
    )
    return pd.DataFrame({'signal': signal})
