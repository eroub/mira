import asyncio
import websockets
import json
import numpy as np
from datetime import datetime

# Global Variables
frequency = 20
bars = []
bar = np.empty(6)
times = np.array([], dtype=np.datetime64)
prices = np.array([], dtype=float)
volumes = np.array([], dtype=float)

async def handle_trade(trade):
    global times, prices, volumes, bar
    # times.append(datetime.strptime(trade['T'], '%Y-%m-%dT%H:%M:%S.%fZ'))
    times = np.append(times, np.datetime64(datetime.fromtimestamp(trade['T']/1000)))
    prices = np.append(prices, trade['p'])
    volumes = np.append(volumes, trade['q'])
    if len(prices) >= frequency:
        # Calculate OHLC and add to numpy array
        prices = np.array(prices, dtype=float)
        ohlc = np.array([prices[0], np.amax(prices), np.amin(prices), prices[-1]])
        bar = np.concatenate((np.array([times[0]], ndmin=1), ohlc, np.array([np.sum(volumes)], dtype=float, ndmin=1)))
        bars.append(bar)
        print(bar)
        if(len(bars) > 3): 
            print(bars)
            exit()
        # Reset indices and arrays storing trade data
        times = np.array([], dtype=np.datetime64)
        prices = np.array([], dtype=float)
        volumes = np.array([], dtype=float)
    return
        
async def trade_stream():
    async with websockets.connect("wss://stream.binance.com:9443/ws/btcusdt@trade") as websocket:
        while True:
            trade = json.loads(await websocket.recv())
            await handle_trade(trade)

asyncio.get_event_loop().run_until_complete(trade_stream())