import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def debug_print(message, DEBUG):
    """Prints debugging messages if DEBUG is True."""
    if DEBUG:
        print(message)


# def fetch_sp500_symbols(include_sectors=False, debug=False):
#     """
#     Fetches S&P 500 symbols from Wikipedia. Optionally includes sectors.

#     Parameters:
#     - include_sectors: Bool, returns symbols with sectors if True.
#     - debug: Bool, prints error message if True and an error occurs.

#     Returns:
#     - List of symbols or list of tuples containing (symbol, sector) based on include_sectors.
#     """
#     try:
#         url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#         tables = pd.read_html(url)
#         sp500_df = tables[0]
#         if include_sectors:
#             symbols_sectors = sp500_df[['Symbol', 'GICS Sector']].values.tolist()
#             return [(symbol, sector) for symbol, sector in symbols_sectors]
#         else:
#             symbols = sp500_df['Symbol'].tolist()
#             return symbols
#     except Exception as e:
#         if debug:
#             print(f"Error fetching S&P 500 symbols: {e}")
#         return []

def fetch_sp500_symbols(include_sectors=False, debug=False):
    """
    Fetches S&P 500 symbols from a hardcoded list. Optionally includes sectors.

    Parameters:
    - include_sectors: Bool, returns symbols with sectors if True.
    - debug: Bool, prints error message if True and an error occurs.

    Returns:
    - List of symbols or list of tuples containing (symbol, sector) based on include_sectors.
    """
    try:
        symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'CRM', 'NXPI', 'ULTA', 'RKLB', 'TWST', 'TSM']

        if include_sectors:
            sectors = ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary', 'Technology', 'Technology', 'Consumer Discretionary', 'Industrials', 'Health Care', 'Technology']
            symbols_sectors = list(zip(symbols, sectors))
            return symbols_sectors
        else:
            return symbols
    except Exception as e:
        if debug:
            print(f"Error fetching S&P 500 symbols: {e}")
        return []


def check_data_exists(symbol, date, collection_name, indicator=None):
    """Checks if data for the given symbol on the specified date and indicator already exists in MongoDB."""
    query = {"symbol": symbol, "date": date}
    if indicator:
        query["indicator"] = indicator
    return db[collection_name].find_one(query) is not None


def get_quarter_dates(year, quarter):
    quarter_month_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    quarter_month_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    start_date = f"{year}-{quarter_month_starts[quarter]}"
    end_date = f"{year}-{quarter_month_ends[quarter]}"
    return start_date, end_date
