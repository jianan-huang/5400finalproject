import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
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


# Load the preprocessed financial news dataset with VADER sentiment labels
news_df = pd.read_csv('data/financial_news_with_sentiment_labels.csv')

# Load the FinBERT model using Hugging Face's transformers library
# Load the FinBERT model using Hugging Face's transformers library
# Set device to 0 to use GPU (if available)
finbert_sentiment_pipeline = pipeline('sentiment-analysis', 
                                      model='yiyanghkust/finbert-tone', 
                                      tokenizer='yiyanghkust/finbert-tone', 
                                      device=0)  # Use GPU if available


# Apply FinBERT sentiment analysis to classify sentiment
def classify_with_finbert(text):
    try:
        result = finbert_sentiment_pipeline(text[:512])  # Using only the first 512 tokens due to input length limitation
        return result[0]['label'].lower()  # 'positive', 'neutral', 'negative'
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {e}")
        return 'neutral'  # Default to 'neutral' if an error occurs

# Apply FinBERT sentiment classification to the dataset
news_df['finbert_sentiment_label'] = news_df['cleaned_content'].astype(str).apply(classify_with_finbert)

# Save the classified dataset to a new CSV file
news_df.to_csv('data/financial_news_with_vader_and_finbert.csv', index=False)

print("Financial news data has been classified with FinBERT sentiment labels and saved to 'financial_news_with_vader_and_finbert.csv'.")


# Load the dataset with VADER and FinBERT sentiment labels
classified_df = pd.read_csv('data/financial_news_with_vader_and_finbert.csv')

# Plot sentiment distributions
plt.figure(figsize=(12, 6))

# VADER Sentiment Distribution
plt.subplot(1, 2, 1)
sns.countplot(x='sentiment_label', data=classified_df, order=['positive', 'neutral', 'negative'])
plt.title('VADER Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# FinBERT Sentiment Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='finbert_sentiment_label', data=classified_df, order=['positive', 'neutral', 'negative'])
plt.title('FinBERT Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('data/vader_vs_finbert_sentiment_distribution.png', format='png', dpi=300)
plt.show()

# Calculate Agreement Between Models
classified_df['agreement'] = classified_df['sentiment_label'] == classified_df['finbert_sentiment_label']
agreement_rate = classified_df['agreement'].mean()
print(f"Agreement Rate between VADER and FinBERT: {agreement_rate:.2f}")

# Inspect a sample of disagreements
disagreements = classified_df[classified_df['agreement'] == False]
print(f"Number of Disagreements: {len(disagreements)}")
print(disagreements[['cleaned_content', 'sentiment_label', 'finbert_sentiment_label']].head(10))





