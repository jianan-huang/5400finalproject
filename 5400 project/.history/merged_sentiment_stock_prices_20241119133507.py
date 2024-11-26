import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import yfinance as yf


# Load the preprocessed financial news dataset
preprocessed_file_path = 'data/preprocessed_financial_news.csv'
news_df = pd.read_csv(preprocessed_file_path)

# Convert the 'date' column to datetime for proper aggregation
news_df['date'] = pd.to_datetime(news_df['date'])

# Assume you have a list of companies and their corresponding stock tickers
company_tickers = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Nvidia": "NVDA",
    "Netflix": "NFLX",
    "Intel": "INTC",
    "IBM": "IBM",
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "Johnson & Johnson": "JNJ",
    "AbbVie": "ABBV",
    "Merck & Co.": "MRK",
    "Amgen": "AMGN",
    "Gilead Sciences": "GILD",
    "Bristol-Myers Squibb": "BMY",
    "Eli Lilly and Company": "LLY",
    "Biogen": "BIIB",
    "JPMorgan Chase": "JPM",
    "Goldman Sachs": "GS",
    "Bank of America": "BAC",
    "Morgan Stanley": "MS",
    "Citigroup": "C",
    "Wells Fargo": "WFC",
    "American Express": "AXP",
    "BlackRock": "BLK",
    "Charles Schwab": "SCHW",
    "Mastercard": "MA"
}

# Step 1: Aggregate Sentiment Scores by Date
# Assume sentiment scores were calculated and are in the 'sentiment_score' column (replace as appropriate)
news_df['sentiment_score'] = news_df['cleaned_content'].apply(lambda x: np.random.uniform(-1, 1))  # Placeholder for actual sentiment score

# Group by company and date to calculate average daily sentiment score
daily_sentiment = news_df.groupby(['company', 'date']).agg({'sentiment_score': 'mean'}).reset_index()

# Ensure that the index of daily_sentiment is reset properly
daily_sentiment.reset_index(drop=True, inplace=True)

# Step 2: Collect Stock Price Data for Each Company
stock_data = []

for company, ticker in company_tickers.items():
    # Download historical stock data for each ticker
    stock_df = yf.download(ticker, start='2023-01-01', end='2024-12-31')
    stock_df.reset_index(inplace=True)
    
    # Rename columns and add a company column
    stock_df = stock_df.rename(columns={"Date": "date"})
    stock_df['company'] = company
    
    # Keep only 'Close' price for analysis and rename it for clarity
    stock_df = stock_df[['company', 'date', 'Close']]
    stock_df = stock_df.rename(columns={"Close": "close_price"})
    
    # Ensure that the index of stock_df is reset properly
    stock_df.reset_index(drop=True, inplace=True)
    
    stock_data.append(stock_df)

# Concatenate all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data, ignore_index=True)

# Debugging: Print columns to verify structure after concatenation
print("Columns of all_stock_data before flattening:", all_stock_data.columns)

# Step 3: Flatten MultiIndex in Columns if Necessary
# If the all_stock_data DataFrame has a MultiIndex in columns, we need to flatten it
if isinstance(all_stock_data.columns, pd.MultiIndex):
    all_stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in all_stock_data.columns]

# Debugging: Print columns to verify structure after flattening
print("Columns of all_stock_data after flattening:", all_stock_data.columns)

# Rename columns back to simpler names if they were flattened
all_stock_data.rename(columns={'company_': 'company', 'date_': 'date'}, inplace=True)

# Ensure the necessary columns are present after renaming
print("Final columns in all_stock_data:", all_stock_data.columns)

# Verification of Index and Data Types
# Check if daily_sentiment has a flat index
print("Index type for daily_sentiment:", daily_sentiment.index)

# Check if all_stock_data has a flat index
print("Index type for all_stock_data:", all_stock_data.index)

# Check columns in daily_sentiment
print("daily_sentiment columns:", daily_sentiment.columns)

# Check columns in all_stock_data
print("all_stock_data columns:", all_stock_data.columns)

# Check the data types of the columns in daily_sentiment
print("Data types in daily_sentiment:\n", daily_sentiment.dtypes)

# Check the data types of the columns in all_stock_data
print("Data types in all_stock_data:\n", all_stock_data.dtypes)

# Convert 'date' column to datetime with the same timezone
# Remove timezone from 'all_stock_data' date column to make it compatible with 'daily_sentiment'
all_stock_data['date'] = pd.to_datetime(all_stock_data['date']).dt.tz_localize(None)

# Convert 'company' column to string if not already
if daily_sentiment['company'].dtype != 'object':
    daily_sentiment['company'] = daily_sentiment['company'].astype(str)
if all_stock_data['company'].dtype != 'object':
    all_stock_data['company'] = all_stock_data['company'].astype(str)

print("Final checks completed. Ready to merge!")

# Step 4: Merge Sentiment Scores and Stock Prices
# Merge daily sentiment scores with stock prices on company and date
merged_df = pd.merge(daily_sentiment, all_stock_data, on=['company', 'date'], how='inner')

df = pd.read_csv('data/merged_sentiment_stock_prices.csv')

# Melt the dataset to convert close price columns into a single column
melted_df = pd.melt(
    df, 
    id_vars=['company', 'date', 'sentiment_score'], 
    value_vars=[col for col in df.columns if col.startswith('close_price_')],
    var_name='close_price_type', 
    value_name='close_price'
)

# Drop rows where 'close_price' is NaN since not all companies have values for all dates
melted_df.dropna(subset=['close_price'], inplace=True)

# Extract the company name from 'close_price_type' and adjust the 'company' column
melted_df['close_price_type'] = melted_df['close_price_type'].str.replace('close_price_', '')
melted_df['company'] = melted_df['close_price_type']

# Drop the 'close_price_type' column as it's no longer needed
melted_df.drop(columns=['close_price_type'], inplace=True)

# Rearrange the columns for better readability
melted_df = melted_df[['company', 'date', 'sentiment_score', 'close_price']]

# Save the transformed dataset
melted_df.to_csv('data/merged_sentiment_stock_prices_long_format.csv', index=False)

print("Dataset has been transformed and saved.")

# Download the necessary VADER data from nltk
nltk.download('vader_lexicon')

# Load the preprocessed financial news dataset
preprocessed_file_path = 'data/preprocessed_financial_news.csv'
news_df = pd.read_csv(preprocessed_file_path)

# Convert the 'date' column to datetime for proper aggregation
news_df['date'] = pd.to_datetime(news_df['date'])

# Instantiate VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Apply VADER to calculate sentiment scores for each cleaned content
news_df['vader_sentiment_score'] = news_df['cleaned_content'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify the sentiment as Positive, Negative, or Neutral based on the compound score
news_df['vader_sentiment'] = news_df['vader_sentiment_score'].apply(
    lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
)

# Save this data to check the results
news_df.to_csv('data/financial_news_with_vader_sentiment.csv', index=False)

# Print to verify that the sentiment columns are added
print(news_df[['cleaned_content', 'vader_sentiment_score', 'vader_sentiment']].head())

# After adding the 'vader_sentiment' column, proceed with aggregation
sentiment_aggregated = news_df.groupby(['company', 'date']).agg({'vader_sentiment': 'first'}).reset_index()

# Load the merged stock and sentiment CSV
merged_csv_path = 'data/merged_sentiment_stock_prices_long_format.csv'
merged_df = pd.read_csv(merged_csv_path)

# Merge sentiment with stock price data
final_df = pd.merge(merged_df, sentiment_aggregated, on=['company', 'date'], how='left')

# Save the final merged dataset
final_df.to_csv('data/final_merged_sentiment_stock_prices.csv', index=False)
print("Final data saved to 'data/final_merged_sentiment_stock_prices.csv'")






























