import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os

nltk.download('punkt_tab')

def create_stopwords():
    """
    Combine English stopwords with finance-specific stopwords.

    Returns:
        set: A set of combined stopwords.
    """
    english_stopwords = set(stopwords.words("english"))

    finance_stopwords = {
        "stock", "market", "price", "share", "profit", "loss", "investment", "investor",
        "financial", "currency", "dollar", "euro", "yen", "revenue", "income"
    }

    return english_stopwords.union(finance_stopwords)


def preprocess_text(text, stop_words, stemmer):
    """
    Preprocess a given text: lowercase, remove punctuation/numbers, tokenize, remove stopwords, and stem.

    Args:
        text (str): The input text to preprocess.
        stop_words (set): Set of stopwords to remove.
        stemmer (PorterStemmer): Stemmer to reduce words to their base form.

    Returns:
        str: The cleaned and preprocessed text.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[\t\n\r]+", " ", text)  # Remove tabs/newlines
    text = re.sub(r"[.,!?;:\"'()\[\]{}<>@#%^&*_+=|\\/~`]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


def filter_by_document_frequency(corpus, min_doc_freq=3):
    """
    Filter words in a corpus based on document frequency.

    Args:
        corpus (list): A list of preprocessed text documents.
        min_doc_freq (int): Minimum document frequency for words to retain.

    Returns:
        list: A filtered list of documents with low-frequency words removed.
    """
    vectorizer = CountVectorizer(min_df=min_doc_freq)
    vectorizer.fit(corpus)
    filtered_corpus = vectorizer.inverse_transform(vectorizer.transform(corpus))
    return [" ".join(doc) for doc in filtered_corpus]


def preprocess_news_articles(input_file, output_file, min_doc_freq=3):
    """
    Load raw news articles, preprocess them, and save the cleaned output.

    Args:
        input_file (str): Path to the raw news articles JSON file.
        output_file (str): Path to save the preprocessed news articles JSON file.
        min_doc_freq (int): Minimum document frequency for words to retain.

    Returns:
        None
    """
    print("Loading data...")
    news_data = pd.read_json(input_file)

    # Initialize stopwords and stemmer
    print("Creating stopwords and initializing stemmer...")
    custom_stopwords = create_stopwords()
    stemmer = PorterStemmer()

    # Preprocess text content
    print("Preprocessing text content...")
    news_data["cleaned_content"] = news_data["content"].apply(
        lambda x: preprocess_text(x, custom_stopwords, stemmer) if isinstance(x, str) else ""
    )

    # Filter by document frequency
    print("Filtering by document frequency...")
    news_data["filtered_content"] = filter_by_document_frequency(
        news_data["cleaned_content"], min_doc_freq=min_doc_freq
    )

    # Save the preprocessed data
    print("Saving preprocessed data...")
    news_data.to_json(output_file, orient="records", date_format="iso")
    print(f"Preprocessing complete! Preprocessed data saved to '{output_file}'.")
