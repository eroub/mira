import os
import asyncio
import websockets
import json
import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import datetime as dt

# Global Variables
frequency = 200
bars = []
bar = np.empty(6)
prices = np.array([], dtype=float)
volumes = np.array([], dtype=float)
times = np.array([], dtype=int)
schema = pa.schema([
        pa.field('id', pa.int64()),
        pa.field('price', pa.float64()),
        pa.field('qty', pa.float64()),
        pa.field('time', pa.int64()),
        pa.field('is_buyer_maker', pa.bool_())
    ])

async def turn_to_bars(trade):
    global times, prices, volumes, bar
    times = np.append(times, np.datetime64(datetime.fromtimestamp(trade['T']/1000)))
    prices = np.append(prices, trade['p'])
    volumes = np.append(volumes, trade['q'])
    if len(prices) >= frequency:
        # Calculate OHLC and add to numpy array
        prices = np.array(prices, dtype=float)
        ohlc = np.array([prices[0], np.amax(prices), np.amin(prices), prices[-1]])
        bar = np.concatenate((np.array([times[0].astype(object)], ndmin=1), ohlc, np.array([np.sum(np.array(volumes, dtype=float))], dtype=float, ndmin=1)))
        bars.append(bar)
        # Reset indices and arrays storing trade data
        times = np.array([], dtype=np.datetime64)
        prices = np.array([], dtype=float)
        volumes = np.array([], dtype=float)
    return

def get_file_name(timestamp):
    date = dt.datetime.fromtimestamp(timestamp / 1000).date()
    return "BTCUSDT-trades-{}-{}-{}".format(date.year, date.month, date.day)
    
async def append_to_parquet(trade):
    global schema
    # Accumulate trade data into dataframe
    trade_data = pd.DataFrame(columns=['id', 'price', 'qty', 'time', 'is_buyer_maker'])
    trade_data = pd.concat([trade_data, pd.DataFrame({'id': [trade['t']], 'price': [trade['p']], 'qty': [trade['q']], 'time': [trade['T']], 'is_buyer_maker': [trade['m']]})], ignore_index=True)
    # Make sure price,qty data is of the right type
    trade_data['price'] = trade_data['price'].astype(float)
    trade_data['qty'] = trade_data['qty'].astype(float)
    # Get filename
    file_name = get_file_name(trade['T']) + ".parquet"
    # If the file exists, append the data to it
    if os.path.exists(file_name):
        existing_df = pd.read_parquet(file_name)
        trade_data = pd.concat([existing_df, trade_data], ignore_index=True)
    trade_data.to_parquet(file_name, compression='snappy')
    trade_data = pd.DataFrame(columns=['id', 'price', 'volume', 'time', 'is_buyer_maker'])
    return
        
async def trade_stream():
    async with websockets.connect("wss://stream.binance.com:9443/ws/btcusdt@trade") as websocket:
        while True:
            trade = json.loads(await websocket.recv())
            await append_to_parquet(trade)

async def start_trade_stream():
    while True:
        try:
            await trade_stream()
        except:
            print("Disconnected. Reconnecting....")
            pass
        # Wait for a while before trying to reconnect
        await asyncio.sleep(1)

asyncio.get_event_loop().run_until_complete(start_trade_stream())