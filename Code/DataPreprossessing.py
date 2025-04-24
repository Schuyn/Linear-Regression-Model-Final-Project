import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def process_timestamp(df, timestamp_col='date'):
    # Convert the date column to datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    # Sort the dataframe based on timestamp
    df.sort_values(by=timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df