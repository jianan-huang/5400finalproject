import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the preprocessed financial news dataset
preprocessed_file_path = 'data/preprocessed_financial_news.csv'
news_df = pd.read_csv(preprocessed_file_path)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply VADER to each row to classify sentiment
def get_vader_sentiment(text):
    if pd.isna(text):
        return 'neutral'
    sentiment_score = sid.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return 'positive'
    elif sentiment_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

news_df['vader_sentiment'] = news_df['cleaned_content'].apply(get_vader_sentiment)

# Save the updated CSV with sentiment
news_df.to_csv('data/preprocessed_sentiment_news.csv', index=False)

