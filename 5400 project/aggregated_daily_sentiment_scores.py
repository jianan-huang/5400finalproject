import pandas as pd
import numpy as np

def process_sentiment_data(input_file, output_file):
    """
    Process financial news sentiment data, calculating daily sentiment scores for all companies.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file with financial news and sentiment labels
    output_file : str
        Path to save the aggregated daily sentiment scores
    """
    # Read the dataset
    news_df = pd.read_csv(input_file)

    # Convert date column to datetime
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date

    # Ensure sentiment scores are numeric
    news_df['vader_sentiment_score'] = pd.to_numeric(news_df['vader_sentiment_score'], errors='coerce').fillna(0)
    news_df['finbert_sentiment_score'] = pd.to_numeric(news_df['finbert_sentiment_score'], errors='coerce').fillna(0)

    # Calculate daily sentiment for all companies
    daily_sentiment_df = news_df.groupby(['company', 'date']).agg(
        vader_daily_sentiment_score=('vader_sentiment_score', 'mean'),
        finbert_daily_sentiment_score=('finbert_sentiment_score', 'mean'),
        news_count=('company', 'count'),  # Number of news articles per company per day
        vader_sentiment_std=('vader_sentiment_score', 'std'),  # Standard deviation of VADER sentiment
        finbert_sentiment_std=('finbert_sentiment_score', 'std')  # Standard deviation of FinBERT sentiment
    ).reset_index()

    # Handle NaN values in the standard deviation columns
    daily_sentiment_df['vader_sentiment_std'] = daily_sentiment_df['vader_sentiment_std'].fillna(0)
    daily_sentiment_df['finbert_sentiment_std'] = daily_sentiment_df['finbert_sentiment_std'].fillna(0)

    # Sort the dataframe by company and date for easier analysis
    daily_sentiment_df = daily_sentiment_df.sort_values(['company', 'date'])

    # Save the complete aggregated dataset
    daily_sentiment_df.to_csv(output_file, index=False)

    # Print summary statistics
    print(f"Total companies processed: {daily_sentiment_df['company'].nunique()}")
    print(f"Total unique dates: {daily_sentiment_df['date'].nunique()}")
    print(f"Date range: {daily_sentiment_df['date'].min()} to {daily_sentiment_df['date'].max()}")
    print(f"Aggregated daily sentiment scores saved to {output_file}")

    # Additional insights
    print("\nSample of aggregated data:")
    print(daily_sentiment_df.head())

    # Optional: Descriptive statistics of sentiment scores
    print("\nDescriptive Statistics:")
    print(daily_sentiment_df[['vader_daily_sentiment_score', 'finbert_daily_sentiment_score']].describe())

# Example usage
process_sentiment_data(
    'data/financial_news_with_vader_and_finbert_sentiment_scores.csv', 
    'data/aggregated_daily_sentiment_scores_full.csv'
)














