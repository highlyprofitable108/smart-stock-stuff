import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.collection_utils import fetch_sp500_symbols, debug_print

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def fetch_company_overview(symbol):
    """
    Fetches company overview information for a given stock symbol from Alpha Vantage.
    """
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        debug_print(f"Failed to fetch data for {symbol}", DEBUG)
        return {}


def store_company_info(company_info):
    """
    Stores or updates company information in the 'company_info' collection.
    """
    if company_info and 'Symbol' in company_info:
        symbol = company_info['Symbol']
        db.company_info.update_one({'Symbol': symbol}, {'$set': company_info}, upsert=True)
        debug_print(f"Stored/Updated company info for {symbol}.", DEBUG)


def main():
    sp500_symbols = fetch_sp500_symbols(debug=DEBUG)
    for symbol in sp500_symbols:
        company_info = fetch_company_overview(symbol)
        if company_info:
            store_company_info(company_info)


if __name__ == "__main__":
    main()
