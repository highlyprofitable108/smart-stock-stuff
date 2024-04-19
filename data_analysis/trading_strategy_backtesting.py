import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import timedelta
import os
import datetime
import random
from dotenv import load_dotenv

load_dotenv()


def load_model(model_path):
    """Load the pre-trained model and its feature names."""
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    return model, feature_names  # Unpack the model and feature names


def load_full_data_from_mongodb(client):
    """Load full data from MongoDB into a DataFrame."""
    db = client['stock_data']
    collection = db['model_data']
    data = pd.DataFrame(list(collection.find()))
    return data


def random_date_interval(data, minimum_days=90, maximum_days=1825):
    """Generate a random start and end date from the dataset ensuring at least a minimum number of days."""
    max_date = pd.to_datetime(data['date'].max())
    start_date = pd.to_datetime(random.choice(data['date']))

    if random.random() < 0.6:  # 60% chance to pick a range between 90 and 180 days
        days = random.randint(minimum_days, 180)
    else:  # 40% chance for longer period up to 5 years
        days = random.randint(181, maximum_days)

    end_date = start_date + timedelta(days=days)
    if end_date > max_date:
        end_date = max_date
        start_date = max(end_date - timedelta(days=days), pd.to_datetime(data['date'].min()))
    return start_date, end_date


def load_data_for_interval(client, start_date, end_date):
    """Load data for a specified date interval from MongoDB."""
    db = client['stock_data']
    collection = db['model_data']
    data = pd.DataFrame(list(collection.find({
        "date": {"$gte": start_date.strftime('%Y-%m-%d'), "$lte": end_date.strftime('%Y-%m-%d')}
    })))
    data['date'] = pd.to_datetime(data['date'])
    return data


def prepare_features(data, feature_names):
    """Prepare features for modeling, consistent with training phase."""
    # Assume RSI is also included in the dataset fetched from MongoDB
    if 'RSI' not in data.columns:
        raise ValueError("RSI data missing from dataset.")

    # Ensure necessary calculations before dropping anything
    if 'volume' in data.columns:
        data['log_volume'] = np.log(data['volume'] + 1)

    # Calculate normalized features
    if 'OBV' in data.columns:
        data['normalized_OBV'] = (data['OBV'] - data['OBV'].mean()) / data['OBV'].std()
    if 'ATR' in data.columns:
        data['normalized_ATR'] = (data['ATR'] - data['ATR'].mean()) / data['ATR'].std()

    IGNORE_LIST = [
        '_id', 'company_name', 'date', 'symbol', 'open', 'high', 'low',
        'Bollinger Bands Lower', 'Bollinger Bands Upper', 'ATR',
        'OBV', 'Stochastic', 'MFI', 'Federal Funds Effective Rate'
    ]

    # Check and extract 'close' before dropping it
    if 'close' in data.columns:
        target = data['close'].copy()
    else:
        raise ValueError("Missing 'close' column in data")

    # Drop ignored columns
    data = data.drop(columns=IGNORE_LIST, errors='ignore')

    # Ensure only the required features are included
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}. Check data preparation steps.")

    features = data[feature_names]

    return features, target


def apply_model_to_data(model, features):
    """Apply the loaded model to generate predictions."""
    predictions = model.predict(features)
    return predictions


def calculate_volatility(prices: pd.Series, window=30) -> pd.Series:
    """Calculate rolling volatility of the given price Series."""
    return prices.pct_change().rolling(window).std() * np.sqrt(window)


def enhanced_trading_strategy(predictions, actual_close, rsi, macd, stochastic, symbols, atr, short_window=10, long_window=30, rsi_threshold=(30, 70), stop_loss_atr_multiplier=3, take_profit_percent=0.10):
    signals = ['hold'] * len(predictions)
    short_ma = actual_close.rolling(window=short_window, min_periods=1).mean()
    long_ma = actual_close.rolling(window=long_window, min_periods=1).mean()
    volatility = calculate_volatility(actual_close)  # Calculate volatility from close prices

    # Dynamic RSI thresholds based on volatility
    dynamic_rsi_lower = rsi_threshold[0] - (volatility * 10)
    dynamic_rsi_upper = rsi_threshold[1] + (volatility * 10)

    last_buy_price = {}

    for i in range(1, len(predictions)):
        symbol = symbols[i]
        current_price = actual_close.iloc[i]
        current_rsi = rsi.iloc[i]
        predicted_diff = predictions[i] - actual_close.iloc[i - 1]
        predicted_diff_percent = predicted_diff / actual_close.iloc[i - 1]

        # Integrate MACD and Stochastic into decision-making
        macd_signal = 'buy' if macd.iloc[i] > 0 and stochastic.iloc[i] < 20 else 'sell' if macd.iloc[i] < 0 and stochastic.iloc[i] > 80 else 'hold'

        # Determine strength of buy/sell signals
        buy_strength = max(0.25, min(1, abs(predicted_diff_percent) * 2))  # More aggressive buying
        sell_strength = max(0.25, min(1, abs(predicted_diff_percent)))  # More conservative selling

        # Trading signals generation based on dynamic conditions and predicted trends
        if (current_rsi < dynamic_rsi_lower.iloc[i] or macd_signal == 'buy') and short_ma.iloc[i] > long_ma.iloc[i] and predicted_diff_percent > 0.01:
            signals[i] = ('buy', buy_strength)
            last_buy_price[symbol] = current_price  # Store the last buy price
        elif (current_rsi > dynamic_rsi_upper.iloc[i] or macd_signal == 'sell') and short_ma.iloc[i] < long_ma.iloc[i] and predicted_diff_percent < -0.01:
            signals[i] = ('sell', sell_strength)

        # Implement dynamic stop-loss and profit-taking
        if symbol in last_buy_price:
            if current_price < last_buy_price[symbol] * (1 - stop_loss_atr_multiplier * atr.iloc[i] / current_price):
                signals[i] = ('sell', sell_strength)  # Sell with conservative strength
            elif current_price > last_buy_price[symbol] * (1 + take_profit_percent):
                signals[i] = ('sell', 1)  # Sell all holdings when profit target is reached

    return signals


def backtest(data, signals, actual_close, dates, symbols, initial_investment=10000, max_buy_percent=0.25, short_term_tax_rate=0.32, long_term_tax_rate=0.15):
    cash = initial_investment
    holdings = {}
    portfolio_value = [cash]
    trade_details = []
    tax_paid = 0
    cooldown = {}

    purchase_dates = {}
    purchase_prices = {}

    for i, (signal, close_price, date, symbol) in enumerate(zip(signals, actual_close, dates, symbols)):
        if isinstance(signal, tuple):
            action, strength = signal
        else:
            action, strength = 'hold', 0

        # Check cooldown period
        if symbol in cooldown and (date - cooldown[symbol]).days < 3:
            continue  # Skip trading if within cooldown period
        else:
            cooldown.pop(symbol, None)  # Remove from cooldown if period is over

        if symbol not in holdings:
            holdings[symbol] = 0
            purchase_dates[symbol] = date

        if action == 'buy' and cash >= close_price:
            max_investment = cash * max_buy_percent * strength
            num_shares = int(max_investment // close_price)
            if num_shares > 0:
                cash -= num_shares * close_price
                holdings[symbol] += num_shares
                purchase_prices[symbol] = close_price
                cooldown[symbol] = date  # Start cooldown period
                trade_details.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'type': 'buy',
                    'shares': num_shares,
                    'price': close_price,
                    'value': num_shares * close_price,
                    'tax_paid': 0
                })

        elif action == 'sell' and holdings[symbol] > 0:
            num_shares_to_sell = int(holdings[symbol] * strength)
            if num_shares_to_sell > 0:
                gain_per_share = close_price - purchase_prices[symbol]
                total_gain = gain_per_share * num_shares_to_sell
                holding_period = (date - purchase_dates[symbol]).days

                tax_rate = long_term_tax_rate if holding_period > 365 else short_term_tax_rate
                tax = total_gain * tax_rate if total_gain > 0 else 0
                tax_paid += tax

                sale_value = num_shares_to_sell * close_price
                cash += sale_value - tax
                holdings[symbol] -= num_shares_to_sell
                cooldown[symbol] = date  # Start cooldown period
                trade_details.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'type': 'sell',
                    'shares': num_shares_to_sell,
                    'price': close_price,
                    'value': sale_value - tax,
                    'tax_paid': tax
                })

        portfolio_value.append(cash + sum(holdings[sym] * actual_close.iloc[i] for sym in holdings))

    last_prices = {sym: data[data['symbol'] == sym].iloc[-1]['close'] for sym in symbols.unique() if sym in holdings}
    final_holdings = {sym: qty for sym, qty in holdings.items() if qty > 0}
    final_portfolio_value = cash + sum(qty * last_prices[sym] for sym, qty in final_holdings.items())
    portfolio_value[-1] = final_portfolio_value

    return portfolio_value, trade_details, final_holdings, last_prices, tax_paid


def get_last_prices(data, symbols):
    """Get the last known prices for each symbol in the holdings."""
    last_prices = {}
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        if not symbol_data.empty:
            last_prices[symbol] = symbol_data.iloc[-1]['close']
    return last_prices


def generate_markdown_report(portfolio_values, trade_details, final_holdings, last_prices, start_date, end_date, tax_paid, filename="data/random_forest/backtest_report.md", initial_investment=10000):
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')
    gross_profit = portfolio_values[-1] - initial_investment
    net_profit = gross_profit - tax_paid  # Calculate net profit after taxes

    # Sort trade details by date
    sorted_trades = sorted(trade_details, key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'))

    md_content = [
        "# Backtest Report",
        f"## Backtest Period: {formatted_start_date} to {formatted_end_date}",
        "## Portfolio Value Over Time",
        "![Portfolio Value](portfolio_plot.png)",
        "## Trades Executed",
        "| Date | Symbol | Type | Shares | Price | Value | Tax Paid |",
        "|------|--------|------|--------|-------|-------|----------|"
    ]

    for trade in sorted_trades:
        tax_info = f"${trade.get('tax_paid', 0):.2f}" if 'tax_paid' in trade else "$0.00"
        md_content.append(f"| {trade['date']} | {trade['symbol']} | {trade['type']} | {trade['shares']} | ${trade['price']:.2f} | ${trade['value']:.2f} | {tax_info} |")

    md_content.append("## Current Holdings")
    for sym, qty in final_holdings.items():
        md_content.append(f"| {sym} | {qty} shares | Last known price: ${last_prices[sym]:.2f} |")

    md_content.append("## Summary Statistics")
    md_content.append(f"Initial Investment: ${initial_investment}")
    md_content.append(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    md_content.append(f"Gross Profit/Loss: ${gross_profit:.2f}")
    md_content.append(f"Net Profit/Loss (after taxes): ${net_profit:.2f}")

    with open(filename, 'w') as file:
        file.write('\n'.join(md_content))
    print(f"Markdown report generated: {filename}")


def plot_results(portfolio_values, filename="data/random_forest/portfolio_plot.png"):
    """Plot the results of the backtest."""
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Backtest Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")


def main():
    mongo_uri = os.getenv('MONGO_URI')
    client = MongoClient(mongo_uri)
    model_path = 'data/random_forest/stock_forecast_model.joblib'
    model, feature_names = load_model(model_path)

    # Load the full dataset from MongoDB
    full_data = load_full_data_from_mongodb(client)

    # Generate a start and end date ensuring a range between 90 days and 5 years
    start_date, end_date = random_date_interval(full_data)

    # Load data specifically for the determined time interval
    data = load_data_for_interval(client, start_date, end_date)

    # Prepare the data for modeling, extracting necessary features and target
    features, actual_close = prepare_features(data, feature_names)
    dates = pd.to_datetime(data['date'])
    symbols = data['symbol']
    rsi = data['RSI']
    macd = data['MACD']
    stochastic = data['Stochastic']
    atr = data['ATR']

    # Apply the predictive model to the prepared features
    predictions = apply_model_to_data(model, features)

    # Generate trading signals based on the predictions and additional indicators
    signals = enhanced_trading_strategy(predictions, actual_close, rsi, macd, stochastic, symbols, atr)

    # Perform backtesting based on the signals to simulate trading over the period
    portfolio_values, trade_details, final_holdings, last_prices, taxes_paid = backtest(data, signals, actual_close, dates, symbols)

    # Plot the results of the portfolio value over time
    plot_results(portfolio_values)

    # Generate a markdown report detailing the trading activity and final statistics
    generate_markdown_report(portfolio_values, trade_details, final_holdings, last_prices, start_date, end_date, taxes_paid)


if __name__ == '__main__':
    main()
