import glob
import pandas as pd

# Find all of the files with the desired file name format
file_names = glob.glob("./daily_data/BTCUSDT-trades-*-*-*.csv")

for file_name in file_names:
    # Read in the data from the file
    data = pd.read_csv(file_name, usecols=[1, 2, 4], names=["price", "volume", "timestamp"])
    # Drop first column if its a header
    if data.iloc[0]['price'] == 'price':
        data = data.drop(0)

    data["price"] = pd.to_numeric(data["price"], errors='coerce', downcast='float')
    data["volume"] = pd.to_numeric(data["volume"], errors='coerce', downcast='float')
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors='coerce', downcast='integer')

