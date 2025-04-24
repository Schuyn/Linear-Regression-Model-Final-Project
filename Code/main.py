'''
author:        Schuyn <98257102+Schuyn@users.noreply.github.com>
date:          2025-04-24 14:39:08
'''
from DataPreprossessing import load_data, process_timestamp
import pandas as pd

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    # Step 1: Load data
    raw_data_path = './Data/nvidia_stock_2015_to_2024.csv'
    df = load_data(raw_data_path)
    
    # Step 2: Timestamp processing
    df = process_timestamp(df, timestamp_col='date')
    
    # You can add more preprocessing here
    
    # Step 3: Save processed data
    processed_data_path = './Data/processed_data.csv'
    save_processed_data(df, processed_data_path)

if __name__ == "__main__":
    main()