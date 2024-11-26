import pandas as pd

# Load the dataset with VADER and FinBERT sentiment labels
news_df = pd.read_csv('data/financial_news_with_vader_and_finbert.csv')

# Convert the 'date' column to datetime format and extract only the date part
news_df['date'] = pd.to_datetime(news_df['date']).dt.date

# Ensure the columns used for aggregation exist
if 'sentiment_label' in news_df.columns and 'finbert_sentiment_label' in news_df.columns:
    # Map sentiment labels to numerical values
    sentiment_mapping = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    # Apply sentiment mapping to create numerical sentiment scores
    news_df['vader_sentiment_score'] = news_df['sentiment_label'].map(sentiment_mapping)
    news_df['finbert_sentiment_score'] = news_df['finbert_sentiment_label'].map(sentiment_mapping)

    # Calculate the average daily sentiment for each company
    daily_sentiment_df = news_df.groupby(['company', 'date']).agg(
        vader_daily_sentiment_score=('vader_sentiment_score', 'mean'),
        finbert_daily_sentiment_score=('finbert_sentiment_score', 'mean')
    ).reset_index()

    # Save the complete aggregated dataset to a new CSV file
    daily_sentiment_df.to_csv('data/aggregated_daily_sentiment_scores_full.csv', index=False)

    print("Aggregated daily sentiment scores for all companies have been calculated and saved.")
else:
    print("Error: Sentiment columns are missing from the dataset.")








