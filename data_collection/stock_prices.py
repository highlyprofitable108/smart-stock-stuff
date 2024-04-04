import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.collection_utils import debug_print, fetch_sp500_symbols, check_data_exists

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def fetch_and_insert_stock_data(symbol):
    try:
        debug_print(f"Fetching daily prices for {symbol}", DEBUG)
        stock_data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.sort_index(inplace=True)

        # Determine the start date: Jan 1, 2019 or the earliest date available for the symbol
        start_date = max(pd.to_datetime('2019-01-01'), stock_data.index.min())
        stock_data = stock_data.loc[start_date:]

        for date, row in stock_data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            if check_data_exists(symbol, date_str, 'stock_prices'):
                debug_print(f"Data for {symbol} on {date_str} already exists.", DEBUG)
                continue

            data_to_insert = {"symbol": symbol, "date": date_str, **row.to_dict()}
            db['stock_prices'].insert_one(data_to_insert)
            debug_print(f"Data for {symbol} on {date_str} inserted.", DEBUG)
            time.sleep(0.2)  # Be mindful of the API's rate limit
    except Exception as e:
        debug_print(f"Error processing {symbol}: {e}", DEBUG)


def main():
    sp500_symbols = fetch_sp500_symbols()
    if not sp500_symbols:
        debug_print("No symbols to process. Exiting...", DEBUG)
        return

    for symbol in sp500_symbols:
        fetch_and_insert_stock_data(symbol)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
