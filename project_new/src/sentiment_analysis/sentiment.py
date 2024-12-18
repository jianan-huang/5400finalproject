import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import os
import logging

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sentiment.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

# Ensure NLTK 'punkt' resource is available
nltk.data.path.append(os.path.expanduser("~/nltk_data"))
try:
    nltk.data.find("tokenizers/punkt")
    logging.info("NLTK 'punkt' tokenizer is available.")
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download("punkt", quiet=True)
    logging.info("Downloaded 'punkt' tokenizer.")


def load_sentiment_words(pos_file_path, neg_file_path):
    """
    Load positive and negative words from text files.

    Parameters:
    pos_file_path (str): Path to the file containing positive words.
    neg_file_path (str): Path to the file containing negative words.

    Returns:
    tuple: Sets of positive and negative words.
    """
    logging.info(f"Loading sentiment word lists from '{pos_file_path}' and '{neg_file_path}'...")
    try:
        with open(pos_file_path, "r") as pos_file:
            positive_words = set(line.strip() for line in pos_file if line.strip())

        with open(neg_file_path, "r") as neg_file:
            negative_words = set(line.strip() for line in neg_file if line.strip())

        logging.info(f"Loaded {len(positive_words)} positive words and {len(negative_words)} negative words.")
        return positive_words, negative_words

    except Exception as e:
        logging.error(f"Error loading sentiment word lists: {e}")
        raise


def calculate_sentiment(text, positive_words, negative_words):
    """
    Calculate sentiment score and polarity of a given text.
    This function calculates the sentiment of a given text using sets of positive and negative words. 

    Parameters:
    text (str): Input text.
    positive_words (set): Set of positive words.
    negative_words (set): Set of negative words.

    Returns:
    tuple: Sentiment score and polarity ('positive' or 'negative').
    """
    try:
        words = word_tokenize(text)

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        score = positive_count - negative_count
        polarity = "positive" if score >= 0 else "negative"

        return score, polarity
    except Exception as e:
        logging.error(f"Error calculating sentiment for text: {e}")
        return 0, "positive"


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
    logging.info("Starting sentiment analysis pipeline...")
    try:
        print("Loading preprocessed data...")
        logging.info(f"Loading preprocessed data from '{input_file}'...")
        news_data = pd.read_json(input_file)

        print("Loading sentiment word lists...")
        positive_words, negative_words = load_sentiment_words(pos_file, neg_file)

        print("Calculating sentiment scores...")
        logging.info("Calculating sentiment scores for each article...")
        news_data["sentiment_score"], news_data["polarity"] = zip(
            *news_data["filtered_content"].apply(
                lambda x: calculate_sentiment(x, positive_words, negative_words)
                if isinstance(x, str)
                else (0, "positive")
            )
        )

        print("Saving sentiment analysis results...")
        logging.info(f"Saving results to '{output_file}'...")
        news_data.to_json(output_file, orient="records", date_format="iso")

        print(f"Sentiment detection complete! Results saved to '{output_file}'.")
        logging.info("Sentiment analysis pipeline complete!")

    except Exception as e:
        logging.error(f"An error occurred during sentiment analysis: {e}")
        raise


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
