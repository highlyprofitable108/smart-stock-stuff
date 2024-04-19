from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from utils.collection_utils import fetch_sp500_symbols, get_quarter_dates

logging.basicConfig(level=logging.DEBUG)
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


def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    if len(prices) < window:
        return pd.Series([np.nan] * len(prices)), pd.Series([np.nan] * len(prices)), pd.Series([np.nan] * len(prices))
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band


def calculate_rsi(prices, window=14):
    if len(prices) < window:
        return pd.Series([np.nan] * len(prices))
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, span1=12, span2=26, signal=9):
    if len(prices) < span2:  # Ensure there is enough data to compute the longer EMA
        return pd.Series([np.nan] * len(prices)), pd.Series([np.nan] * len(prices))

    exp1 = prices.ewm(span=span1, adjust=False).mean()
    exp2 = prices.ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    return macd, signal_line


def calculate_ema(prices, window=20):
    if len(prices) < window:
        return pd.Series([np.nan] * len(prices))
    return prices.ewm(span=window, adjust=False).mean()


def calculate_stochastic_oscillator(high, low, close, window=14):
    if len(close) < window:
        return pd.Series([np.nan] * len(close))
    l14 = low.rolling(window=window).min()
    h14 = high.rolling(window=window).max()
    return 100 * (close - l14) / (h14 - l14)


def calculate_atr(high, low, close, window=14):
    if len(close) < window:
        return pd.Series([np.nan] * len(close))
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()


def calculate_obv(close, volume):
    if len(close) < 2:
        return pd.Series([np.nan] * len(close))
    return volume.where(close.diff() > 0, -volume).cumsum()


def calculate_mfi(high, low, close, volume, window=14):
    if len(close) < window:
        return pd.Series([np.nan] * len(close))

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price.diff() > 0).fillna(0).rolling(window=window).sum()
    negative_flow = money_flow.where(typical_price.diff() < 0).fillna(0).rolling(window=window).sum()

    # Check if negative_flow is zero anywhere to prevent division by zero
    zero_negative_flow = (negative_flow == 0).any()

    if zero_negative_flow:
        return pd.Series([100.0] * len(close))  # Assigning a default value when negative flow is zero
    else:
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))


def fetch_technical_indicators(symbol, start_date, end_date, lookback_period=45):
    # Adjust the start date to fetch additional historical data for calculations
    adjusted_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_period)
    price_data = pd.DataFrame(list(db.stock_prices.find({
        'symbol': symbol,
        'date': {'$gte': adjusted_start_date.strftime('%Y-%m-%d'), '$lte': end_date}
    })))

    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data.set_index('date', inplace=True)
    price_data.sort_index(inplace=True)

    if not price_data.empty:
        close = price_data['4. close'].astype(float)
        high = price_data['2. high'].astype(float)
        low = price_data['3. low'].astype(float)
        volume = price_data['6. volume'].astype(float)

        ema = calculate_ema(close).iloc[-1]
        stochastic = calculate_stochastic_oscillator(high, low, close).iloc[-1]
        atr = calculate_atr(high, low, close).iloc[-1]
        obv = calculate_obv(close, volume).iloc[-1]
        mfi = calculate_mfi(high, low, close, volume).iloc[-1]
        bollinger_mean, bollinger_upper, bollinger_lower = calculate_bollinger_bands(close)
        rsi = calculate_rsi(close).iloc[-1]
        macd, signal_line = calculate_macd(close)

        indicators = {
            'EMA': ema,
            'Stochastic': stochastic,
            'ATR': atr,
            'OBV': obv,
            'MFI': mfi,
            'Bollinger Bands Mean': bollinger_mean.iloc[-1],
            'Bollinger Bands Upper': bollinger_upper.iloc[-1],
            'Bollinger Bands Lower': bollinger_lower.iloc[-1],
            'RSI': rsi,
            'MACD': macd.iloc[-1],
            'MACD Signal Line': signal_line.iloc[-1]
        }
        return indicators
    else:
        return {}


def generate_model_data(start_date, end_date):
    sp500_symbols = fetch_sp500_symbols()
    print(f"Generating model data for S&P 500 symbols from {start_date} to {end_date}")

    for symbol in sp500_symbols:
        print(f"Processing symbol: {symbol}")
        
        # Fetch enough historical data to accurately calculate indicators
        # Assuming a lookback period of 30 days is enough to cover all indicators
        historical_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=30)
        historical_end_date = pd.to_datetime(end_date)

        price_data = pd.DataFrame(list(db.stock_prices.find({
            'symbol': symbol,
            'date': {'$gte': historical_start_date.strftime('%Y-%m-%d'), '$lte': historical_end_date.strftime('%Y-%m-%d')}
        })))

        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data.set_index('date', inplace=True)
        price_data.sort_index(inplace=True)

        if price_data.empty:
            print(f"No historical price data available for {symbol} from {historical_start_date.strftime('%Y-%m-%d')} to {historical_end_date.strftime('%Y-%m-%d')}")
            continue

        # Iterate through each business day in the requested date range
        for single_date in pd.date_range(start_date, end_date, freq='B'):
            date_str = single_date.strftime('%Y-%m-%d')

            # Ensure there is data for this specific date
            if date_str in price_data.index:
                day_data = price_data.loc[date_str]
                technical_indicators = fetch_technical_indicators(symbol, date_str, date_str)

                model_data = {
                    'symbol': symbol,
                    'date': date_str,
                    'open': day_data['1. open'],
                    'high': day_data['2. high'],
                    'low': day_data['3. low'],
                    'close': day_data['4. close'],
                    'adjusted_close': day_data['5. adjusted close'],
                    'volume': day_data['6. volume'],
                    **fetch_economic_indicators(date_str),
                    **technical_indicators  # Add technical indicators directly
                }

                company_info = db.company_info.find_one({'Symbol': symbol})
                if company_info:
                    model_data['company_name'] = company_info.get('Name', '')

                filter_query = {'symbol': symbol, 'date': date_str}
                update_query = {'$set': model_data}
                db.model_data.update_one(filter_query, update_query, upsert=True)
            else:
                print(f"No price data for {symbol} on {date_str}")

        print(f"Finished processing {symbol}")

    print("Model data generation completed.")


if __name__ == "__main__":
    start_date = '2019-01-01'
    end_date = '2023-12-31'
    generate_model_data(start_date, end_date)
