import os
import time
import pandas as pd
from alpha_vantage.techindicators import TechIndicators
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.collection_utils import debug_print, fetch_sp500_symbols, check_data_exists

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def insert_indicator_data(symbol, date_str, indicator_name, value):
    """Helper function to insert indicator data into MongoDB."""
    data_to_insert = {
        "symbol": symbol,
        "date": date_str,
        "indicator": indicator_name,
        "value": value
    }
    db['technical_indicators'].insert_one(data_to_insert)
    debug_print(f"{indicator_name} data for {symbol} on {date_str} inserted.", DEBUG)


def fetch_and_insert_indicator_data(symbol, indicator, function):
    """Fetches and inserts indicator data for the given symbol."""
    try:
        debug_print(f"Fetching {indicator} for {symbol}", DEBUG)
        data, _ = function(symbol=symbol)  # Unpack the tuple directly for the data frame

        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)

        start_date = max(pd.to_datetime('2019-01-01'), data.index.min())
        data = data.loc[start_date:]

        for date, row in data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            if indicator == "MACD":
                # For MACD, handle each component separately
                for col in row.index:
                    indicator_name = f"{col}"
                    if not check_data_exists(symbol, date_str, 'technical_indicators', indicator=indicator_name):
                        insert_indicator_data(symbol, date_str, indicator_name, row[col])
                        time.sleep(0.2)  # Be mindful of the API's rate limit
                    else:
                        debug_print(f"Skipping processing {symbol} for {indicator} as data already exists on {date_str}", DEBUG)
            else:
                # For SMA, RSI, or any other single-value indicators
                value = row[indicator] if indicator in row else None
                if value is not None and not check_data_exists(symbol, date_str, 'technical_indicators', indicator=indicator):
                    insert_indicator_data(symbol, date_str, indicator, value)
                    time.sleep(0.2)  # Be mindful of the API's rate limit
                else:
                    debug_print(f"Skipping processing {symbol} for {indicator} as data already exists on {date_str}", DEBUG)

    except Exception as e:
        debug_print(f"Error processing {symbol} for {indicator}: {e}", DEBUG)
        import traceback
        traceback.print_exc()


def main():
    try:
        sp500_symbols = fetch_sp500_symbols()
        if not sp500_symbols:
            debug_print("No symbols to process. Exiting...", DEBUG)
            return

        indicators = {
            "SMA": lambda symbol: ti.get_sma(symbol=symbol, interval='daily', series_type='close'),
            "RSI": lambda symbol: ti.get_rsi(symbol=symbol, interval='daily', series_type='close'),
            "MACD": lambda symbol: ti.get_macd(symbol=symbol, interval='daily', series_type='close')
        }

        for symbol in sp500_symbols:
            for indicator, function in indicators.items():
                fetch_and_insert_indicator_data(symbol, indicator, function)

    except Exception as e:
        debug_print(f"An error occurred: {e}", DEBUG)
        import traceback
        traceback.print_exc()
    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()
