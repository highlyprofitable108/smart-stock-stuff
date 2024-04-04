from fredapi import Fred
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from utils.collection_utils import debug_print

load_dotenv()

FRED_API_KEY = os.getenv('FRED_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')

fred = Fred(api_key=FRED_API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def fetch_and_store_fred_data(series_id, series_name):
    """Fetch a data series from FRED and store it in MongoDB."""
    data = fred.get_series(series_id, observation_start='2018-12-01')

    collection = db['economic_indicators']
    for date, value in data.items():
        document = {
            'date': date.strftime('%Y-%m-%d'),
            'indicator': series_name,
            'value': value
        }
        collection.update_one({'date': document['date'], 'indicator': series_name},
                              {'$set': document}, upsert=True)
    debug_print(f"Data for {series_name} inserted into MongoDB.", True)


def main():
    indicators = {
        'GDP': 'Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
        'FEDFUNDS': 'Federal Funds Effective Rate'
    }

    for series_id, series_name in indicators.items():
        fetch_and_store_fred_data(series_id, series_name)


if __name__ == "__main__":
    main()
