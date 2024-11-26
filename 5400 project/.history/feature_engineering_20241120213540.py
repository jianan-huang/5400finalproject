import pandas as pd
import numpy as np

def feature_engineering(stock_data_file, sentiment_data_file, output_file):
    # Load historical stock price data
    stock_df = pd.read_csv(stock_data_file)

    # Convert date column to datetime
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date

    # Calculate historical price features
    # Daily Return = (Today's Close Price - Yesterday's Close Price) / Yesterday's Close Price
    stock_df['daily_return'] = stock_df.groupby('company')['close_price'].pct_change()

    # Volatility: Rolling standard deviation of daily returns over a window of 7 days
    stock_df['volatility'] = stock_df.groupby('company')['daily_return'].rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)

    # Moving Averages: 7-day and 30-day moving averages of close price
    stock_df['moving_avg_7'] = stock_df.groupby('company')['close_price'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    stock_df['moving_avg_30'] = stock_df.groupby('company')['close_price'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

    # Load daily sentiment scores
    sentiment_df = pd.read_csv(sentiment_data_file)

    # Convert date column to datetime
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

    # Merge historical price features with daily sentiment scores
    combined_df = pd.merge(stock_df, sentiment_df, on=['company', 'date'], how='left')

    # Create lagged sentiment scores (e.g., 1-day lagged scores)
    combined_df['vader_daily_sentiment_score_lag1'] = combined_df.groupby('company')['vader_daily_sentiment_score'].shift(1)
    combined_df['finbert_daily_sentiment_score_lag1'] = combined_df.groupby('company')['finbert_daily_sentiment_score'].shift(1)

    # Instead of filling all missing values with zero, use a more nuanced approach:
    combined_df['daily_return'].fillna(0, inplace=True)  # Fill missing daily returns with zero (e.g., first day)
    combined_df['volatility'].fillna(method='bfill', inplace=True)  # Use backward fill for volatility
    combined_df['moving_avg_7'].fillna(method='bfill', inplace=True)  # Backward fill for moving averages
    combined_df['moving_avg_30'].fillna(method='bfill', inplace=True)

    # Fill sentiment scores using a forward fill, as sentiment can often carry over day-to-day
    combined_df['vader_daily_sentiment_score'].fillna(method='ffill', inplace=True)
    combined_df['finbert_daily_sentiment_score'].fillna(method='ffill', inplace=True)
    combined_df['vader_daily_sentiment_score_lag1'].fillna(method='ffill', inplace=True)
    combined_df['finbert_daily_sentiment_score_lag1'].fillna(method='ffill', inplace=True)

    # Save the final combined dataset
    combined_df.to_csv(output_file, index=False)

    # Print summary statistics
    print(f"Feature engineered data saved to {output_file}")
    print(f"Total rows: {combined_df.shape[0]}")
    print("\nSample of combined data:")
    print(combined_df.head())

# Example usage
feature_engineering(
    'data/historical_stock_prices.csv', 
    'data/aggregated_daily_sentiment_scores_full.csv',
    'data/feature_engineered_data_updated.csv'
)

