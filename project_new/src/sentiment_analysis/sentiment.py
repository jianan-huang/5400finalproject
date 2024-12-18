import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure NLTK 'punkt' resource is available
nltk.data.path.append(os.path.expanduser("~/nltk_data"))
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def load_sentiment_words(pos_file_path, neg_file_path):
    """
    Load positive and negative words from text files.

    Parameters:
    pos_file_path (str): Path to the file containing positive words.
    neg_file_path (str): Path to the file containing negative words.

    Returns:
    tuple: Sets of positive and negative words.
    """
    with open(pos_file_path, "r") as pos_file:
        positive_words = set(line.strip() for line in pos_file if line.strip())

    with open(neg_file_path, "r") as neg_file:
        negative_words = set(line.strip() for line in neg_file if line.strip())

    return positive_words, negative_words


def calculate_sentiment(text, positive_words, negative_words):
    """
    Calculate sentiment score and polarity of a given text.

    Parameters:
    text (str): Input text.
    positive_words (set): Set of positive words.
    negative_words (set): Set of negative words.

    Returns:
    tuple: Sentiment score and polarity ('positive' or 'negative').
    """
    words = word_tokenize(text)

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    score = positive_count - negative_count
    polarity = "positive" if score >= 0 else "negative"

    return score, polarity


def calculate_sentiments(input_file, pos_file, neg_file, output_file):
    """
    Load preprocessed data, calculate sentiment scores, and save results.

    Parameters:
    input_file (str): Path to the preprocessed news articles JSON file.
    pos_file (str): Path to the positive words file.
    neg_file (str): Path to the negative words file.
    output_file (str): Path to save the sentiment analysis results.

    Returns:
    None
    """
    print("Loading preprocessed data...")
    news_data = pd.read_json(input_file)

    print("Loading sentiment word lists...")
    positive_words, negative_words = load_sentiment_words(pos_file, neg_file)

    print("Calculating sentiment scores...")
    news_data["sentiment_score"], news_data["polarity"] = zip(
        *news_data["filtered_content"].apply(
            lambda x: calculate_sentiment(x, positive_words, negative_words)
            if isinstance(x, str)
            else (0, "positive")
        )
    )

    print("Saving sentiment analysis results...")
    news_data.to_json(output_file, orient="records", date_format="iso")
    print(f"Sentiment detection complete! Results saved to '{output_file}'.")


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "data/preprocessed_news_articles.json"
    POS_FILE = "data/positive-words.txt"
    NEG_FILE = "data/negative-words.txt"
    OUTPUT_FILE = "data/news_with_sentiments.json"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Perform sentiment analysis
    calculate_sentiments(INPUT_FILE, POS_FILE, NEG_FILE, OUTPUT_FILE)
