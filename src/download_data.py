import os
import requests
import zipfile
import io
import pandas as pd
from config import RAW_DATA_DIR

def download_binance_data(symbol="BTCUSDT", interval="1m", year="2024", months=["01", "02"]):
    """
    Downloads historical klines from Binance Vision, unzips them, 
    and combines them into a single CSV.
    """
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    
    all_dfs = []
    
    for month in months:
        file_name = f"{symbol}-{interval}-{year}-{month}.zip"
        url = f"{base_url}/{symbol}/{interval}/{file_name}"
        
        print(f"Downloading {file_name}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Read zip file from memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # The zip contains a single CSV file with the same name
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # Binance CSVs don't have headers by default
                    columns = [
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ]
                    df = pd.read_csv(f, names=columns)
                    all_dfs.append(df)
            print(f"Successfully loaded {len(df)} rows for {year}-{month}.")
        else:
            print(f"Failed to download {url}. Status: {response.status_code}")
            
    if all_dfs:
        print("Combining datasets...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by timestamp just in case
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        output_path = RAW_DATA_DIR / f"{symbol}_{interval}_sample.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined data to {output_path}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No data was downloaded.")

if __name__ == "__main__":
    # Downloading Jan and Feb 2024 to keep it lightweight for MVP dev
    download_binance_data(year="2024", months=["01", "02", "03"])
