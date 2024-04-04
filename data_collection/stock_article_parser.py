import os
import requests
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.collection_utils import debug_print, get_quarter_dates

# Load environment variables
load_dotenv()

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def data_exists_for_quarter(symbol, from_date, to_date):
    """Check if data already exists for the given symbol and quarter."""
    quarter_range = f"{from_date} to {to_date}"
    return db.sentiment_txt.find_one({
        'symbol': symbol,
        'quarter': quarter_range
    }) is not None


def fetch_news_articles(company_name, from_date, to_date):
    """Fetches news articles for a given company within a specified date range from GNews."""
    url = "https://gnews.io/api/v4/search"
    params = {
        'q': '"' + company_name + '"',
        'from': from_date,
        'to': to_date,
        'token': GNEWS_API_KEY,
        'lang': 'en',
        'max': 10  # Limit to top 10 articles per quarter
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [(article['title'], article['publishedAt']) for article in articles]
    else:
        debug_print(f"Failed to fetch articles for {company_name}: {response.status_code} - {response.text}", DEBUG)
        return []


def store_article_info(symbol, articles, from_date, to_date):
    """Stores article titles and publication dates in the sentiment_txt collection."""
    for title, publishedAt in articles:
        document = {
            'symbol': symbol,
            'title': title,
            'date': publishedAt[:10],
            'quarter': f"{from_date} to {to_date}"
        }
        db.sentiment_txt.insert_one(document)
        debug_print(f"Stored article for {symbol}: {title}", DEBUG)


def main(start_year=2019, start_quarter=1):
    loop_count = 0
    sp500_symbols = db.company_info.distinct("Symbol")
    end_year = datetime.now().year
    end_quarter = (datetime.now().month - 1) // 3 + 1

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if year == end_year and quarter > end_quarter:
                break
            if year == start_year and quarter < start_quarter:
                continue

            from_date, to_date = get_quarter_dates(year, quarter)

            for symbol in sp500_symbols:
                if data_exists_for_quarter(symbol, from_date, to_date):
                    debug_print(f"Data already exists for {symbol} in {from_date} to {to_date}", DEBUG)
                    continue

                company_name = db.company_info.find_one({"Symbol": symbol}, {"Name": 1}).get('Name')
                if company_name:
                    articles = fetch_news_articles(company_name, from_date, to_date)
                    if articles:
                        store_article_info(symbol, articles, from_date, to_date)

                loop_count += 1
                if loop_count >= 5000:
                    print("Rate limit reached for the day, to get more data please wait until API reset.")
                    return


if __name__ == "__main__":
    main()
