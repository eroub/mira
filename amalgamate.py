import glob
import pandas as pd
import pyarrow
from helpers.preprocessing import convert_to_tick
from tqdm import tqdm

# Set desired frequency (num trades to make a tick bar)
frequency = 1000

# Find all of the files with the desired file name format, then sort them
file_names = sorted(glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv"))
# Initialize an empty dataframe
res = pd.DataFrame()
# Loop through the selected file names
for file_name in tqdm(file_names):
    # Read in the data from the file
    try:
        # Read in the data from the file
        data = pd.read_csv(file_name, usecols=[1, 2, 4], names=["price", "volume", "timestamp"], low_memory=False)
        # Drop first column if its a header
        if data.iloc[0]['price'] == 'price':
            data = data.drop(0)
        # Drop any rows with missing values
        data.dropna(inplace=True)
        # Coerce odd values to be of the correct type
        data["price"] = pd.to_numeric(data["price"], errors='coerce', downcast='float')
        data["volume"] = pd.to_numeric(data["volume"], errors='coerce', downcast='float')
        data["timestamp"] = pd.to_numeric(data["timestamp"], errors='coerce', downcast='integer')
    except:
        # If the above fails for whatever reasons there is something wrong with the file, print it
        print(file_name)
        continue
    # Call convert_to_tick for each file
    subset_data = convert_to_tick(data, frequency)
    # Append the data to the array
    res = pd.concat([res, subset_data], axis=0)

print(res)
# Create dataxxx.parquet
res.to_parquet("data{}.parquet".format(frequency))