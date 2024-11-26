import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the necessary NLTK VADER resource
nltk.download('vader_lexicon')

# Load the preprocessed financial news dataset
news_df = pd.read_csv('data/preprocessed_financial_news.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis to classify sentiment
def classify_sentiment(text):
    sentiment = sid.polarity_scores(text)
    compound_score = sentiment['compound']
    
    # Classify as Positive, Negative, or Neutral based on compound score
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Classify each row in the dataset
news_df['sentiment_label'] = news_df['cleaned_content'].astype(str).apply(classify_sentiment)

# Save the classified dataset to a new CSV file
news_df.to_csv('data/financial_news_with_sentiment_labels.csv', index=False)

print("Financial news data has been classified with sentiment labels and saved to 'financial_news_with_sentiment_labels.csv'.")
