import pandas as pd
import itertools

# Load the dataset with VADER and FinBERT sentiment labels
news_df = pd.read_csv('data/financial_news_with_vader_and_finbert.csv')

# Convert the 'date' column to datetime and extract the date part
news_df['date'] = pd.to_datetime(news_df['date']).dt.date

# Mapping sentiment labels to numerical scores
sentiment_mapping = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

# Apply the mapping to create numerical sentiment scores
news_df['vader_sentiment_score'] = news_df['sentiment_label'].map(sentiment_mapping)
news_df['finbert_sentiment_score'] = news_df['finbert_sentiment_label'].map(sentiment_mapping)

# Calculate the average daily sentiment for each company
daily_sentiment_df = news_df.groupby(['company', 'date']).agg(
    vader_daily_sentiment_score=('vader_sentiment_score', 'mean'),
    finbert_daily_sentiment_score=('finbert_sentiment_score', 'mean')
).reset_index()

# Ensure all companies and dates are represented
# Create a date range covering all possible dates
start_date = news_df['date'].min()
end_date = news_df['date'].max()
all_dates = pd.date_range(start=start_date, end=end_date).date

# Create a complete grid of companies and dates
all_companies = news_df['company'].unique()
full_grid = pd.DataFrame(list(itertools.product(all_companies, all_dates)), columns=['company', 'date'])

# Merge the full grid with the aggregated sentiment scores
full_daily_sentiment_df = pd.merge(full_grid, daily_sentiment_df, on=['company', 'date'], how='left')

# Fill missing values for sentiment scores with the sector average or a default value
# Here, we fill with 0, but you could use the sector's average sentiment score
full_daily_sentiment_df['vader_daily_sentiment_score'].fillna(0, inplace=True)
full_daily_sentiment_df['finbert_daily_sentiment_score'].fillna(0, inplace=True)

# Save the complete aggregated dataset to a new CSV
full_daily_sentiment_df.to_csv('data/aggregated_daily_sentiment_scores.csv', index=False)

print("Complete average daily sentiment scores have been calculated and saved.")







