import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import yfinance as yf


# Load the preprocessed financial news dataset
preprocessed_file_path = 'data/preprocessed_financial_news.csv'
news_df = pd.read_csv(preprocessed_file_path)

# Convert the 'date' column to datetime for proper aggregation
news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
news_df.dropna(subset=['date'], inplace=True)  # Remove rows with invalid dates

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
news_df['sentiment_score'] = news_df['cleaned_content'].apply(lambda x: np.random.uniform(-1, 1))  # Placeholder for actual sentiment score

# Group by company and date to calculate average daily sentiment score
daily_sentiment = news_df.groupby(['company', 'date']).agg({'sentiment_score': 'mean'}).reset_index()
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

    # Debug: Print columns to verify renaming
    print(f"Columns for {company} after renaming: {stock_df.columns}")

    # Keep only 'Close' price for analysis and rename it for clarity
    stock_df = stock_df[['company', 'date', 'Close']]
    stock_df = stock_df.rename(columns={"Close": "close_price"})
    
    stock_data.append(stock_df)

# Concatenate all stock data into a single DataFrame
all_stock_data = pd.concat(stock_data, ignore_index=True)

# Debugging: Print columns to verify structure after concatenation
print("Columns of all_stock_data before flattening:", all_stock_data.columns)

# Step 3: Flatten MultiIndex in Columns if Necessary
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
if 'date' in all_stock_data.columns:
    all_stock_data['date'] = pd.to_datetime(all_stock_data['date']).dt.tz_localize(None)
else:
    print("Error: 'date' column not found in all_stock_data.")

# Convert 'company' column to string if not already
if daily_sentiment['company'].dtype != 'object':
    daily_sentiment['company'] = daily_sentiment['company'].astype(str)
if all_stock_data['company'].dtype != 'object':
    all_stock_data['company'] = all_stock_data['company'].astype(str)

print("Final checks completed. Ready to merge!")

# Step 4: Merge Sentiment Scores and Stock Prices
merged_df = pd.merge(daily_sentiment, all_stock_data, on=['company', 'date'], how='outer')

# Save the transformed dataset
merged_df.to_csv('data/merged_sentiment_stock_prices_long_format.csv', index=False)

print("Dataset has been transformed and saved.")


# Load the CSV file
file_path = 'data/merged_sentiment_stock_prices_long_format.csv'
df = pd.read_csv(file_path)

# Melt the dataframe to convert close price columns into a single column
melted_df = pd.melt(
    df, 
    id_vars=['company', 'date', 'sentiment_score'], 
    value_vars=[col for col in df.columns if col.startswith('close_price_')],
    var_name='close_price_type', 
    value_name='close_price'
)

# Extract the company name from 'close_price_type' and adjust the 'company' column
melted_df['close_price_type'] = melted_df['close_price_type'].str.replace('close_price_', '')
melted_df['company'] = melted_df['close_price_type']

# Drop the 'close_price_type' column as it's no longer needed
melted_df.drop(columns=['close_price_type'], inplace=True)

# Fill missing values in the 'close_price' and 'sentiment_score' columns with 'N/A'
melted_df['close_price'].fillna('N/A', inplace=True)
melted_df['sentiment_score'].fillna('N/A', inplace=True)

# Rearrange the columns for better readability
melted_df = melted_df[['company', 'date', 'sentiment_score', 'close_price']]

# Save the transformed dataset
melted_df.to_csv('data/converted_merged_sentiment_stock_prices.csv', index=False)

print("Dataset has been transformed and saved.")





























