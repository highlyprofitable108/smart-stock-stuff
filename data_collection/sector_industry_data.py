from pymongo import MongoClient
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.collection_utils import fetch_sp500_symbols, debug_print

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def calculate_daily_returns():
    """
    Calculate the daily return for each stock and update the documents in the 'stock_prices' collection.
    Daily return = (close - open) / open
    """
    cursor = db.stock_prices.find({})
    for doc in cursor:
        if '1. open' in doc and '4. close' in doc:
            daily_return = (doc['4. close'] - doc['1. open']) / doc['1. open'] if doc['1. open'] != 0 else 0
            db.stock_prices.update_one({'_id': doc['_id']}, {'$set': {'daily_return': daily_return}})


def aggregate_sector_performance(date):
    """
    Aggregate performance metrics for each sector for a specific date.
    """
    sp500_symbols = fetch_sp500_symbols(include_sectors=True, debug=DEBUG)
    sector_keys = {symbol: sector for symbol, sector in sp500_symbols}

    sectors = set(sector_keys.values())
    sector_performance = {}

    for sector in sectors:
        symbols_in_sector = [symbol for symbol, sec in sector_keys.items() if sec == sector]

        pipeline = [
            {"$match": {"symbol": {"$in": symbols_in_sector}, "date": date.strftime('%Y-%m-%d')}},
            {"$group": {
                "_id": "$symbol",
                "avg_daily_return": {"$avg": "$daily_return"},
                "total_volume": {"$sum": "$6. volume"}
            }}
        ]
        results = db.stock_prices.aggregate(pipeline)

        # Process aggregation results
        returns = [result["avg_daily_return"] for result in results if "avg_daily_return" in result]
        volumes = [result["total_volume"] for result in results if "total_volume" in result]

        if returns and volumes:
            avg_return = sum(returns) / len(returns)
            volatility = (max(returns) - min(returns)) / avg_return if avg_return != 0 else 0
            max_gain = max(returns)
            min_gain = min(returns)
            total_volume = sum(volumes)

            sector_performance[sector] = {
                'avg_return': avg_return,
                'volatility': volatility,
                'max_gain': max_gain,
                'min_gain': min_gain,
                'total_volume': total_volume
            }

    return sector_performance


def store_sector_performance(sector_performance, date):
    """
    Store the aggregated sector performance metrics.
    """
    for sector, metrics in sector_performance.items():
        document = {
            'date': date.strftime('%Y-%m-%d'),
            'sector': sector,
            **metrics
        }
        db['sector_performance'].update_one({'date': document['date'], 'sector': sector},
                                            {'$set': document}, upsert=True)
        debug_print(f"Stored sector performance for {sector} on {date}.", DEBUG)


def main(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        sector_performance = aggregate_sector_performance(current_date)
        if sector_performance:
            store_sector_performance(sector_performance, current_date)
        current_date += timedelta(days=1)


if __name__ == "__main__":
    start_date = datetime(2019, 1, 1)
    end_date = datetime.now()
    main(start_date, end_date)
