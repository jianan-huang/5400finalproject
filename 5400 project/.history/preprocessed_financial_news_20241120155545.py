import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download only necessary NLTK components
nltk.download('punkt')
nltk.download('stopwords')

# Load each CSV file
finance_df = pd.read_csv('data/finance_financial_news_10_companies.csv')
healthcare_df = pd.read_csv('data/healthcare_financial_news_10_companies.csv')
tech_df = pd.read_csv('data/tech_financial_news_10_companies.csv')

# Combine the dataframes
combined_df = pd.concat([finance_df, healthcare_df, tech_df], ignore_index=True)

# Drop rows with missing 'content' or 'published_at' to avoid errors
combined_df.dropna(subset=['content', 'published_at'], inplace=True)

# Text Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

# Apply text preprocessing to the 'content' column
combined_df['content'] = combined_df['content'].fillna('')  # Fill NaN with empty strings
combined_df['cleaned_content'] = combined_df['content'].astype(str).apply(preprocess_text)

# Data Alignment: Create a 'date' column from the 'published_at' column
combined_df['published_at'] = pd.to_datetime(combined_df['published_at'], errors='coerce')
combined_df.dropna(subset=['published_at'], inplace=True)  # Drop rows where date conversion failed
combined_df['date'] = combined_df['published_at'].dt.date

# Save the preprocessed dataset to a new CSV file
combined_df.to_csv('data/preprocessed_financial_news.csv', index=False)

print("Preprocessed dataset has been saved to 'data/preprocessed_financial_news.csv'.")

