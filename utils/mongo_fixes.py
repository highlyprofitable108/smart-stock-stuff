import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Function to calculate the quarter from a date
def get_quarter_from_date(date_str):
    year, month, _ = map(int, date_str.split('-'))
    quarter = ((month - 1) // 3) + 1
    quarter_start_date = datetime(year, 3 * quarter - 2, 1).strftime('%Y-%m-%d')
    if quarter == 4:
        quarter_end_date = datetime(year, 12, 31).strftime('%Y-%m-%d')
    else:
        quarter_end_date = datetime(year, 3 * quarter + 1, 1) - timedelta(days=1)
        quarter_end_date = quarter_end_date.strftime('%Y-%m-%d')
    return f"{quarter_start_date} to {quarter_end_date}"


# MongoDB connection setup
MONGO_URI = os.getenv('MONGO_URI')  # Get MongoDB URI from environment variable
client = MongoClient(MONGO_URI)
db = client.stock_data


def update_quarters_in_db():
    articles = db.sentiment_txt.find()  # Find all documents

    for article in articles:
        correct_quarter = get_quarter_from_date(article['date'])
        # Update the document with the correct quarter
        db.sentiment_txt.update_one(
            {'_id': article['_id']},
            {'$set': {'quarter': correct_quarter}}
        )
        print(f"Updated article {article['_id']} to quarter {correct_quarter}")


if __name__ == '__main__':
    update_quarters_in_db()
