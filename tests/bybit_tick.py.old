import os
import numpy as np
import fbm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime
from pybit import usdt_perpetual
from time import sleep

load_dotenv()
API_KEY=os.getenv("API_KEY")
API_SECRET=os.getenv("API_SECRET")

ws_linear = usdt_perpetual.WebSocket(
    test=False,
    ping_interval=30,  # the default is 30
    ping_timeout=10,  # the default is 10
    domain="bybit"  # the default is "bybit"
)

# Global Variables
frequency = 10
it = 0
times = []
prices = []
volumes = []
res = np.zeros(shape=(1,), dtype=[('date', 'datetime64[ms]'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')])

def handle_message(message):
    global it, times, prices, volumes, res
    trades = message["data"]
    new_trades = [[trade["timestamp"], float(trade["price"]), trade["size"]] for trade in trades]
    for trade in new_trades:
        timestamp = trade[0]
        price = trade[1]
        size = trade[2]
        datetime_object = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        times.append(datetime_object)
        prices.append(price)
        volumes.append(size)
    if len(prices) >= frequency:
        for i in range(frequency, len(prices), frequency):
            res = np.resize(res, (it+1,))
            res[it][0] = times[i-1]                        # time
            res[it][1] = prices[i-frequency]               # open
            res[it][2] = np.max(prices[i-frequency:i])     # high
            res[it][3] = np.min(prices[i-frequency:i])     # low
            res[it][4] = prices[i-1]                       # close
            res[it][5] = np.sum(volumes[i-frequency:i])    # volume
            it += 1
        it = 0
        times.clear()
        prices.clear()
        volumes.clear()
    return handle_message

# Generate FBM samples
hurst_exponent = 0.5
fbm_samples = fbm.fbm(n=len(close_prices), hurst=hurst_exponent, length=len(close_prices), method='daviesharte')

# Subscribe to BTCUSDT
ws_linear.trade_stream(handle_message, "BTCUSDT")

while True:
    sleep(1)
