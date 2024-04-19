import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from utils.collection_utils import debug_print

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')

# MongoDB connection setup
client = MongoClient(MONGO_URI)
db = client.stock_data


def get_quarter_dates(year, quarter):
    """
    Returns the start and end dates of a given quarter in a given year.
    """
    quarter_month_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    quarter_month_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    start_date = f"{year}-{quarter_month_starts[quarter]}"
    end_date = f"{year}-{quarter_month_ends[quarter]}"
    return start_date, end_date


def cleanup_sentiment_data(symbol, from_date, to_date):
    """
    Removes articles and sentiment scores for the specified symbol and quarter
    if the sentiment score does not exist or is 0.
    """
    quarter_range = f"{from_date} to {to_date}"
    sentiment_score = db.sentiment_scores.find_one({
        'symbol': symbol,
        'quarter': quarter_range
    })

    if sentiment_score is None or sentiment_score.get('average_sentiment_score', 1) == 0:
        db.sentiment_txt.delete_many({'symbol': symbol, 'quarter': quarter_range})
        db.sentiment_scores.delete_one({'symbol': symbol, 'quarter': quarter_range})
        debug_print(f"Removed articles and score for {symbol} in quarter {quarter_range}.", True)
    else:
        debug_print(f"Valid score exists for {symbol} in quarter {quarter_range}. No action taken.", True)


def remove_duplicate_articles(symbol, from_date, to_date):
    """
    Removes duplicate articles for the specified symbol within a date range,
    keeping only the first document encountered for each unique combination
    of symbol and article.
    """
    pipeline = [
        {
            "$match": {
                "symbol": symbol,
                "date": {"$gte": from_date, "$lte": to_date}
            }
        },
        {
            "$group": {
                "_id": {"symbol": "$symbol", "article": "$article"},
                "docIds": {"$push": "$_id"}
            }
        },
        {
            "$match": {
                "docIds": {"$not": {"$size": 1}}
            }
        }
    ]

    duplicates = list(db.sentiment_txt.aggregate(pipeline))

    for duplicate in duplicates:
        # Keep the first document and remove the rest
        to_remove = duplicate["docIds"][1:]  # Skip the first ID
        db.sentiment_txt.delete_many({"_id": {"$in": to_remove}})


if __name__ == '__main__':
    symbols = db.company_info.distinct("Symbol")
    for symbol in symbols:
        # Iterates through each year and quarter
        for year in range(2019, datetime.now().year + 1):
            for quarter in range(1, 5):
                from_date, to_date = get_quarter_dates(year, quarter)
                # Skip future quarters
                if datetime.strptime(from_date, '%Y-%m-%d') > datetime.now():
                    continue
                cleanup_sentiment_data(symbol, from_date, to_date)
                remove_duplicate_articles(symbol, from_date, to_date)
