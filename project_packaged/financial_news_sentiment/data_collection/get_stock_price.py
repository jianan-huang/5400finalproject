import yfinance as yf
import pandas as pd
import os

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

def fetch_stock_prices(start_date, end_date, output_csv_path='data/raw/historical_stock_prices.csv'):
    # Create an empty list to store the dataframes for each company
    stock_data = []

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Loop through each company to collect its stock price data
    for company, ticker in company_tickers.items():
        print(f"Collecting data for {company} ({ticker})...")
        
        try:
            # Download historical stock data
            stock_df = yf.download(ticker, start=start_date, end=end_date)
            
            # Reset index to make 'Date' a column
            stock_df.reset_index(inplace=True)
            
            # Add company name to the data
            stock_df['company'] = company
            
            # Keep only relevant columns (Date and Close price)
            stock_df = stock_df[['company', 'Date', 'Close']]
            
            # Rename columns to match the previous expectation
            stock_df.columns = ['company', 'date', 'close_price']
            
            # Append to the list
            stock_data.append(stock_df)
        
        except Exception as e:
            print(f"Error collecting data for {company}: {e}")

    # Concatenate all individual company dataframes into one
    all_stock_data = pd.concat(stock_data, ignore_index=True)

    # Drop any rows with missing close prices
    all_stock_data.dropna(subset=['close_price'], inplace=True)

    # Save the final dataset to CSV
    all_stock_data.to_csv(output_csv_path, index=False)
    print(f"Stock price data saved to '{output_csv_path}'.")

# Example usage
if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    fetch_stock_prices(start_date, end_date)
