import pandas as pd
import os

import pandas as pd
import os

def preprocess_data(news_csv_path, stock_csv_path, output_csv_path):
    # Load the news data
    print("Loading news data...")
    news_df = pd.read_csv(news_csv_path)

    # Parse 'published_at' to datetime format
    news_df['published_at'] = pd.to_datetime(news_df['published_at']).dt.date
    news_df.rename(columns={'published_at': 'date'}, inplace=True)

    # Drop any duplicates in the news data
    news_df.dropna(subset=['content'], inplace=True)
    print(f"Loaded {len(news_df)} news articles.")

    # Load the stock price data
    print("Loading stock price data...")
    stock_df = pd.read_csv(stock_csv_path)

    # Parse 'date' to datetime format in the stock data
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date


    # Handling missing dates in stock data for each company
    print("Filling missing dates in stock price data...")
    all_companies = stock_df['company'].unique()
    complete_stock_data = []

    for company in all_companies:
        company_data = stock_df[stock_df['company'] == company].copy()

        # Create a complete date range from the min to max date in the company's data
        full_date_range = pd.date_range(start=company_data['date'].min(), end=company_data['date'].max())

        # Set the date as the index to align it with the full date range
        company_data.set_index('date', inplace=True)

        # Reindex the DataFrame to the full date range, keeping the existing data and adding missing rows
        company_data = company_data.reindex(full_date_range, method='ffill')

        # Reset the index and rename the new date index column
        company_data.reset_index(inplace=True)
        company_data.rename(columns={'index': 'date'}, inplace=True)

        # Add back the company name in the newly added rows
        company_data['company'] = company

        complete_stock_data.append(company_data)

    # Concatenate the data for all companies
    stock_df = pd.concat(complete_stock_data, ignore_index=True)

    print(f"Filled stock price data has {len(stock_df)} entries.")

    # Merge news and stock price data on 'company' and 'date'
    print("Merging news and stock data...")
    merged_df = pd.merge(news_df, stock_df, on=['company', 'date'], how='inner')

    # Drop rows with missing values, if any, after merging
    merged_df.dropna(inplace=True)
    print(f"Merged dataset has {len(merged_df)} records.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save the processed data to CSV
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to '{output_csv_path}'.")

# Example usage
if __name__ == "__main__":
    news_csv_path = 'data/raw/Financial_news.csv'
    stock_csv_path = 'data/raw/historical_stock_prices.csv'
    output_csv_path = 'data/processed/Processed_merged_data.csv'

    preprocess_data(news_csv_path, stock_csv_path, output_csv_path)


# Example usage
if __name__ == "__main__":
    news_csv_path = 'data/raw/Financial_news.csv'
    stock_csv_path = 'data/raw/historical_stock_prices.csv'
    output_csv_path = 'data/processed/Processed_merged_data.csv'
    
    preprocess_data(news_csv_path, stock_csv_path, output_csv_path)
