import yfinance as yf
import pandas as pd

# Define a dictionary of companies and their corresponding stock tickers
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

# Define start and end dates for data collection
start_date = "2023-01-01"
end_date = "2024-12-31"

# Create an empty list to store the dataframes for each company
stock_data = []

# Loop through each company to collect its stock price data
for company, ticker in company_tickers.items():
    print(f"Collecting data for {company} ({ticker})...")
    
    # Download historical stock data
    stock_df = yf.download(ticker, start=start_date, end=end_date)
    stock_df.reset_index(inplace=True)
    
    # Add company name to the data
    stock_df['company'] = company
    
    # Keep only relevant columns (Date and Close price)
    stock_df = stock_df[['company', 'Date', 'Close']]
    stock_df.rename(columns={'Date': 'date', 'Close': 'close_price'}, inplace=True)
    
    # Append to the list
    stock_data.append(stock_df)

# Concatenate all individual company dataframes into one
all_stock_data = pd.concat(stock_data, ignore_index=True)

# Save the reshaped DataFrame to a CSV file
output_file_path = 'data/historical_stock_prices_long_format.csv'  # Replace with your desired output path
all_stock_data.to_csv(output_file_path, index=False)
print(f"Data has been reshaped and saved to '{output_file_path}'.")




