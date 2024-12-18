import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

# Ensure the logs directory exists before configuring logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "timeseries.log"),  # Save log in the 'logs' directory
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)


def generate_timeseries_plot(sentiment_file, stock_file, output_file):
    """
    Generate a time series plot comparing news sentiment scores and stock prices.

    Args:
        sentiment_file (str): Path to the JSON file containing sentiment scores.
        stock_file (str): Path to the JSON file containing stock prices.
        output_file (str): Path to save the generated time series plot.

    Returns:
        None
    """
    logging.info("Starting time series plot generation...")

    try:
        # Load sentiment data
        logging.info(f"Loading sentiment data from '{sentiment_file}'...")
        sentiment_data = pd.read_json(sentiment_file)

        # Load stock data
        logging.info(f"Loading stock data from '{stock_file}'...")
        stock_data = pd.read_json(stock_file)

        # Preprocess dates and sentiment scores
        logging.info("Preprocessing data...")
        sentiment_data['published_at'] = pd.to_datetime(sentiment_data['published_at'])
        sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'].astype(float)

        # Aggregate sentiment scores by date
        daily_sentiment = sentiment_data.groupby(sentiment_data['published_at'].dt.date)['sentiment_score'].mean()
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment.columns = ['Date', 'Average Sentiment Score']

        # Convert dates to datetime
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)

        # Merge stock data and sentiment data
        logging.info("Merging sentiment and stock price data...")
        merged_data = pd.merge(stock_data, daily_sentiment, on='Date', how='inner')

        # Plot time series
        logging.info("Generating time series plot...")
        plt.figure(figsize=(14, 8))

        # Plot stock prices
        plt.plot(merged_data['Date'], merged_data['Adj Close'], label='Adjusted Close Price', color='blue')

        # Scale sentiment scores to match the range of stock prices
        scaled_sentiment = (merged_data['Average Sentiment Score'] - merged_data['Average Sentiment Score'].min()) / (
            merged_data['Average Sentiment Score'].max() - merged_data['Average Sentiment Score'].min()
        ) * merged_data['Adj Close'].max()

        # Plot scaled sentiment scores
        plt.plot(merged_data['Date'], scaled_sentiment, label='Scaled Sentiment Score', color='orange', linestyle='--')

        # Customize plot
        plt.title("News Sentiment vs. Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()

        # Save the plot
        plt.savefig(output_file)
        logging.info(f"Time series plot saved to '{output_file}'.")

        plt.close()
        logging.info("Time series plot generation complete.")

    except Exception as e:
        logging.error(f"An error occurred during time series plot generation: {e}")
        raise


if __name__ == "__main__":
    # File paths
    SENTIMENT_FILE = "data/news_with_sentiments.json"
    STOCK_FILE = "data/stock_prices.json"
    OUTPUT_FILE = "data/news_sentiment_vs_stock_prices.png"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Generate time series plot
    generate_timeseries_plot(SENTIMENT_FILE, STOCK_FILE, OUTPUT_FILE)
