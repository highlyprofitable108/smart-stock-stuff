from utils.collection_utils import debug_print, get_quarter_dates
import os
import requests
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = True

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data


def fetch_news_articles(company_name, from_date, to_date):
    """Fetches news articles for a given company within a specified date range from GNews."""
    url = "https://gnews.io/api/v4/search"
    params = {
        'q': '"' + company_name + '"',
        'from': from_date,
        'to': to_date,
        'token': GNEWS_API_KEY,
        'lang': 'en',
        'max': 10  # Attempt to fetch up to 10 articles
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [(article['title'], article['publishedAt']) for article in articles]
    else:
        debug_print(f"Failed to fetch articles for {company_name}: {response.status_code} - {response.text}", DEBUG)
        return []


def get_quarter_from_date(date):
    """Calculate the quarter for a given date."""
    year, month, _ = [int(part) for part in date.split('-')]
    quarter = (month - 1) // 3 + 1
    return year, quarter


def store_article_info(symbol, articles, from_date, to_date):
    """Stores article titles and publication dates in the sentiment_txt collection, ensuring uniqueness."""
    for title, publishedAt in articles:
        # Check if an article with the same title (or title start) and symbol already exists
        if db.sentiment_txt.count_documents({'symbol': symbol, 'title': {"$regex": title[:20]}}) == 0:
            pub_year, pub_quarter = get_quarter_from_date(publishedAt[:10])
            pub_from_date, pub_to_date = get_quarter_dates(pub_year, pub_quarter)
            document = {
                'symbol': symbol,
                'title': title,
                'date': publishedAt[:10],
                'quarter': f"{pub_from_date} to {pub_to_date}"
            }
            db.sentiment_txt.insert_one(document)
            debug_print(f"Stored article for {symbol}: {title}", DEBUG)
        else:
            debug_print(f"Duplicate article skipped for {symbol}: {title}", DEBUG)


def main(start_year=2019, start_quarter=1):
    sp500_symbols = db.company_info.distinct("Symbol")
    end_year = datetime.now().year
    end_quarter = (datetime.now().month - 1) // 3 + 1

    for year in range(start_year, end_year + 1):
        for quarter in range(start_quarter if year == start_year else 1, end_quarter if year == end_year else 5):
            from_date, to_date = get_quarter_dates(year, quarter)

            for symbol in sp500_symbols:
                company_name = db.company_info.find_one({"Symbol": symbol}, {"Name": 1}).get('Name')
                if company_name:
                    articles = fetch_news_articles(company_name, from_date, to_date)
                    if articles:
                        store_article_info(symbol, articles, from_date, to_date)


if __name__ == "__main__":
    main()
