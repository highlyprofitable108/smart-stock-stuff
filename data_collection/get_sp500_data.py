import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def debug_print(message, debug):
    """Utility function to print messages if debugging is enabled."""
    if debug:
        print(message)


def fetch_and_insert_index_data(symbol):
    try:
        debug_print(f"Fetching daily prices for {symbol}", DEBUG)
        # Note: Alpha Vantage may not support adjusted close for indices directly, adjust accordingly
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')

        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)

        start_date = max(pd.to_datetime('2019-01-01'), data.index.min())
        data = data.loc[start_date:]

        for date, row in data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            if db['index_prices'].find_one({"symbol": symbol, "date": date_str}):
                debug_print(f"Data for {symbol} on {date_str} already exists.", DEBUG)
                continue

            data_to_insert = {"symbol": symbol, "date": date_str, **row.to_dict()}
            db['index_prices'].insert_one(data_to_insert)
            debug_print(f"Data for {symbol} on {date_str} inserted.", DEBUG)
            time.sleep(12)  # API limit for free Alpha Vantage accounts is usually 5 calls per minute
    except Exception as e:
        debug_print(f"Error processing {symbol}: {e}", DEBUG)


def main():
    # S&P 500 index symbol, verify the correct symbol with your data provider
    sp500_index_symbol = 'GSPC'
    fetch_and_insert_index_data(sp500_index_symbol)


if __name__ == "__main__":
    main()
