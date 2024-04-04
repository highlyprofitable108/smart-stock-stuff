from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from utils.collection_utils import fetch_sp500_symbols, get_quarter_dates

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def fetch_sentiment_score(symbol, date):
    """
    Fetches sentiment scores for a given symbol and date from MongoDB.
    """
    current_date = datetime.strptime(date, '%Y-%m-%d')
    current_quarter = (current_date.month - 1) // 3 + 1
    current_year = current_date.year
    while True:
        quarter_str = f"Q{current_quarter}"
        start_date, end_date = get_quarter_dates(current_year, current_quarter)
        if current_date >= datetime.strptime(start_date, '%Y-%m-%d'):
            break
        current_year -= 1

    cursor = db.sentiment_scores.find({'symbol': symbol, 'quarter': quarter_str})
    scores = [document['average_sentiment_score'] for document in cursor]
    if scores:
        return sum(scores) / len(scores)
    else:
        return None


def fetch_economic_indicators(date):
    """
    Fetches the most recent economic indicators for a given symbol and date from MongoDB.
    """
    indicators = {
        "Consumer Price Index for All Urban Consumers: All Items": None,
        "Federal Funds Effective Rate": None,
        "Gross Domestic Product": None,
        "Unemployment Rate": None
    }

    for indicator_name in indicators.keys():
        cursor = db.economic_indicators.find({
            'indicator': indicator_name,
            'date': {'$lte': date}
        }).sort('date', -1).limit(1)

        for document in cursor:
            indicators[indicator_name] = document['value']

    return indicators


def fetch_technical_indicators(symbol, date):
    """
    Fetches technical indicators for a given symbol and date from MongoDB.
    """
    cursor = db.technical_indicators.find({'symbol': symbol, 'date': date})
    indicators = {}
    for document in cursor:
        indicator = document['indicator']
        value = document['value']
        indicators[indicator] = value
    return indicators


def generate_model_data(start_date, end_date):
    """
    Generates model data by combining stock prices, technical indicators,
    economic indicators, sentiment scores, and company information.
    """
    sp500_symbols = fetch_sp500_symbols()

    for symbol in sp500_symbols:
        stock_prices = db.stock_prices.find({'symbol': symbol, 'date': {'$gte': start_date, '$lte': end_date}})

        for price in stock_prices:
            date = price['date']

            open_price = price['1. open']
            high_price = price['2. high']
            low_price = price['3. low']
            close_price = price['4. close']
            adjusted_close = price['5. adjusted close']
            volume = price['6. volume']

            sentiment_score = fetch_sentiment_score(symbol, date)

            economic_indicators = fetch_economic_indicators(date)

            technical_indicators = fetch_technical_indicators(symbol, date)

            company_info = db.company_info.find_one({'Symbol': symbol})
            if company_info:
                company_name = company_info.get('Name', '')

            model_data = {
                'symbol': symbol,
                'company_name': company_name,
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adjusted_close': adjusted_close,
                'volume': volume,
                'sentiment_score': sentiment_score,
                'economic_indicators': economic_indicators,
                'technical_indicators': technical_indicators
            }

            filter_query = {'symbol': symbol, 'date': date}
            update_query = {'$set': model_data}
            db.model_data.update_one(filter_query, update_query, upsert=True)


if __name__ == "__main__":
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    generate_model_data(start_date, end_date)
