import datetime
import random
import pandas as pd
import os

def convert_to_tick(data, frequency):
    # Group the dataframe by every 'frequency' rows
    df_grouped = data.groupby(data.index // frequency)

    # Compute the open, high, low, close, volume, and timestamp for each group
    df_agg = df_grouped.agg({'price': ['first', 'max', 'min', 'last'],
                            'volume': 'sum',
                            'timestamp': 'last'})
    # Rename the columns
    df_agg.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    # Reset the index
    df_agg.reset_index(drop=True, inplace=True)
    # Return df_agg
    return df_agg

def grab_data(sample_file_names, frequency):
    # Initialize an empty dataframe
    res = pd.DataFrame()
    # Loop through the selected file names
    for file_name in sample_file_names:
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
    return res

def process_from_parquet(num_days, num_check_days, frequency=200):
    # Read in the parquet file
    df = pd.read_parquet("{}/helpers/data{}.parquet".format(os.getcwd(), frequency))
    df = df.sort_values(by='timestamp')
    df = df.reset_index(drop=True)

        # Get the max_index allowed by leaving enough room for the num_check_days
    last_timestamp = df['timestamp'].iloc[-1]
    max_timestamp = last_timestamp - num_check_days * 86400 * 1000
    # Find the max_index by searching for the first timestamp that is greater than max_timestamp
    for i, timestamp in enumerate(df['timestamp'].values):
        if timestamp > max_timestamp:
            max_index = i - 1
            break
    random_index = random.randint(0, min(max_index, len(df) - 1))

    # Get the timestamp of the random index
    random_timestamp = df.loc[random_index, 'timestamp']
    
    # Filter the dataframe based on timestamp for model_data
    end_timestamp = random_timestamp + num_days*86400*1000
    model_data = df.loc[(df['timestamp'] >= random_timestamp) & (df['timestamp'] < end_timestamp),:]

    # Filter the dataframe based on timestamp for check_data
    start_timestamp = end_timestamp
    end_timestamp = start_timestamp + num_check_days*86400*1000
    check_data = df.loc[(df['timestamp'] >= start_timestamp) & (df['timestamp'] < end_timestamp),:]

    return model_data, check_data

def process_from_parquet_step(num_days, num_check_days, num_steps, frequency=200):
    # Read in the parquet file
    df = pd.read_parquet("{}/helpers/data{}.parquet".format(os.getcwd(), frequency))
    df = df.sort_values(by='timestamp')
    df = df.reset_index(drop=True)

    # Get the max_index allowed by leaving enough room for the num_check_days
    last_timestamp = df['timestamp'].iloc[-1]
    max_timestamp = last_timestamp - num_check_days * 86400 * 1000
    # Find the max_index by searching for the first timestamp that is greater than max_timestamp
    for i, timestamp in enumerate(df['timestamp'].values):
        if timestamp > max_timestamp:
            max_index = i - 1
            break
    random_index = random.randint(0, min(max_index, len(df) - 1))
    
    results = []
    for i in range(num_steps):
        # Get the timestamp of the random index
        random_timestamp = df.loc[random_index, 'timestamp']

        # Filter the dataframe based on timestamp for model_data
        end_timestamp = random_timestamp + num_days*86400*1000
        model_data = df.loc[(df['timestamp'] >= random_timestamp) & (df['timestamp'] < end_timestamp),:]

        # Filter the dataframe based on timestamp for check_data
        start_timestamp = end_timestamp
        end_timestamp = start_timestamp + num_check_days*86400*1000
        check_data = df.loc[(df['timestamp'] >= start_timestamp) & (df['timestamp'] < end_timestamp),:]

        results.append((model_data, check_data))
        random_index += 1

    return results
