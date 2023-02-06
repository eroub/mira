import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# Graph variables
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

def create_plot():
    global res
    # Plotting res['close']
    fig, ax = plt.subplots()
    line, = ax.plot(res['date'], res['close'], lw=2)
    plt.title("Closing Price")
    plt.xlabel("Time")
    plt.ylabel("Price (USDT)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

# Show chart
def update_plot(num):
    line.set_data(times[:num], res['close'][:num])
    ax.relim()
    ax.autoscale_view()
    return line

def handle_trade(message):
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

        print(res)
        # Plotting res['close']
        create_plot()
        
        it = 0
        times.clear()
        prices.clear()
        volumes.clear()

def run_ws():
    ws_linear.start()

def stop_ws():
    ws_linear.stop()

if __name__ == '__main__':
    # Subscribe to BTCUSDT
    ws_linear.trade_stream(handle_trade, "BTCUSDT")
    plt.show()