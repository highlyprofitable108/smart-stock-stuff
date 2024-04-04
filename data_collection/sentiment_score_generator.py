import os
import nltk
from pymongo import MongoClient
from dotenv import load_dotenv
from utils.collection_utils import get_quarter_dates, fetch_sp500_symbols
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.stock_data
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """Analyzes the sentiment of a given piece of text using VADER."""
    score = sia.polarity_scores(text)
    return score['compound']  # Returns the compound score


def process_articles_for_symbol(symbol, start_year=2019, end_year=datetime.now().year):
    """Processes articles for a specific symbol and computes average sentiment scores by quarter."""
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):  # Iterate through quarters
            from_date, to_date = get_quarter_dates(year, quarter)
            quarter_range = f"{from_date} to {to_date}"

            # Skip quarters that are in the future
            if datetime.strptime(to_date, '%Y-%m-%d') > datetime.now():
                continue

            # Check if sentiment scores for this quarter and symbol already exist
            if db.sentiment_scores.find_one({'symbol': symbol, 'quarter': quarter_range}):
                continue  # Skip processing if already done

            cursor = db.sentiment_txt.find({'symbol': symbol, 'quarter': quarter_range})
            scores = [analyze_sentiment(doc['title']) for doc in cursor if doc.get('title')]

            if scores:
                avg_score = sum(scores) / len(scores)
                # Store or update the average score for this symbol and quarter
                db.sentiment_scores.update_one(
                    {'symbol': symbol, 'quarter': quarter_range},
                    {'$set': {'average_sentiment_score': avg_score}},
                    upsert=True
                )


def process_articles(start_year=2019):
    """Initiates processing of articles for all S&P 500 symbols."""
    sp500_symbols = fetch_sp500_symbols()  # Ensure this fetches a list of symbols
    for symbol in sp500_symbols:
        process_articles_for_symbol(symbol, start_year=start_year)


if __name__ == "__main__":
    process_articles()
