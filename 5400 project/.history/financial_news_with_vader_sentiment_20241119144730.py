import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary VADER lexicon data
nltk.download('vader_lexicon')

# Load the preprocessed financial news dataset
preprocessed_file_path = 'data/preprocessed_financial_news.csv'
news_df = pd.read_csv(preprocessed_file_path)

# Convert the 'date' column to datetime for proper aggregation
news_df['date'] = pd.to_datetime(news_df['date'])

# Instantiate VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Ensure all values in 'cleaned_content' are strings, replace NaN with an empty string
news_df['cleaned_content'] = news_df['cleaned_content'].fillna('').astype(str)

# Apply VADER to calculate sentiment scores for each cleaned content
news_df['vader_sentiment_score'] = news_df['cleaned_content'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify the sentiment as Positive, Negative, or Neutral based on the compound score
news_df['vader_sentiment'] = news_df['vader_sentiment_score'].apply(
    lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
)

# Save the dataset with VADER sentiment scores and labels
financial_news_with_vader_sentiment_path = 'data/financial_news_with_vader_sentiment.csv'
news_df.to_csv(financial_news_with_vader_sentiment_path, index=False)

print("Dataset with VADER sentiment scores has been saved.")
