import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load preprocessed financial news dataset
news_df = pd.read_csv('data/preprocessed_financial_news.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis to the cleaned content
def get_sentiment_score(text):
    sentiment = sid.polarity_scores(text)
    return sentiment['compound']

news_df['sentiment_score'] = news_df['cleaned_content'].astype(str).apply(get_sentiment_score)

# Data Alignment: Create a 'date' column from the 'published_at' column if not already created
news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize(None)

# Calculate daily sentiment score for each company
# Here we take the average sentiment score per company per day
daily_sentiment_df = news_df.groupby(['company', 'date'])['sentiment_score'].mean().reset_index()
daily_sentiment_df.rename(columns={'sentiment_score': 'daily_sentiment_score'}, inplace=True)

# Load historical stock prices dataset
stock_prices_df = pd.read_csv('data/historical_stock_prices.csv')

# Convert the 'date' column to datetime format without timezone
stock_prices_df['date'] = pd.to_datetime(stock_prices_df['date']).dt.tz_localize(None)

# Align the stock price data with the daily sentiment scores
aligned_df = pd.merge(stock_prices_df, daily_sentiment_df, on=['company', 'date'], how='left')

# Save the aligned dataset to a new CSV file
aligned_df.to_csv('data/aligned_stock_prices_sentiment.csv', index=False)

print("Stock price data has been aligned with daily sentiment scores and saved.")






