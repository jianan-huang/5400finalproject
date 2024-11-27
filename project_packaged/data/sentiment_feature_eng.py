import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import nltk

nltk.download('vader_lexicon')

def calculate_sentiment_score(text):
    # Initialize VADER sentiment intensity analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_dict = sid.polarity_scores(text)
    return sentiment_dict['compound']

def sentiment_analysis_and_feature_engineering(input_csv_path, output_csv_path):
    # Load the merged data
    print("Loading merged dataset...")
    merged_df = pd.read_csv(input_csv_path)

    # Calculate sentiment score for each news article
    print("Calculating sentiment scores...")
    merged_df['sentiment_score'] = merged_df['content'].apply(calculate_sentiment_score)

    # Feature Engineering
    print("Performing feature engineering...")
    
    # Convert 'date' to datetime format
    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # Sort the dataset by company and date
    merged_df.sort_values(by=['company', 'date'], inplace=True)

    # Calculate lagged sentiment scores (previous day's sentiment score for each company)
    merged_df['lagged_sentiment_score'] = merged_df.groupby('company')['sentiment_score'].shift(1)

    # Moving average of sentiment scores over a window of 3 days for each company
    merged_df['sentiment_moving_avg_3d'] = merged_df.groupby('company')['sentiment_score'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Calculate daily price returns for each company
    merged_df['daily_return'] = merged_df.groupby('company')['close_price'].pct_change()

    # Calculate 5-day moving average of the close price for each company
    merged_df['price_moving_avg_5d'] = merged_df.groupby('company')['close_price'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Calculate 5-day rolling volatility (standard deviation of close price)
    merged_df['volatility_5d'] = merged_df.groupby('company')['close_price'].transform(lambda x: x.rolling(window=5, min_periods=1).std())

    # Calculate change in sentiment score compared to the previous day for each company
    merged_df['sentiment_change'] = merged_df.groupby('company')['sentiment_score'].diff()

    # Drop rows with NaN values after feature engineering
    merged_df.dropna(inplace=True)

    # Create the target column: 1 if daily return is positive, 0 otherwise
    merged_df['target'] = merged_df['daily_return'].apply(lambda x: 1 if x > 0 else 0)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save the dataset ready for modeling
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Feature-engineered data saved to '{output_csv_path}'.")

# Example usage
if __name__ == "__main__":
    input_csv_path = 'data/processed/Processed_merged_data.csv'
    output_csv_path = 'data/processed/Feature_engineered_data.csv'
    
    sentiment_analysis_and_feature_engineering(input_csv_path, output_csv_path)
