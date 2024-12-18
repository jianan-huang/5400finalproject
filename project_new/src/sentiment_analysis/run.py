import os
from preprocessing import preprocess_news_articles
from sentiment import calculate_sentiments
from tfidf import generate_tfidf
from model_eval import load_data, train_and_evaluate_models, plot_confusion_matrices, save_results
from visualization import generate_word_clouds, generate_sentiment_distribution

# File paths
RAW_DATA_FILE = "data/news_articles_with_full_content.json"
PREPROCESSED_DATA_FILE = "data/preprocessed_news_articles.json"
POSITIVE_WORDS_FILE = "data/positive-words.txt"
NEGATIVE_WORDS_FILE = "data/negative-words.txt"
SENTIMENT_OUTPUT_FILE = "data/news_with_sentiments.json"
TFIDF_CSV_FILE = "data/tfidf_representation.csv"
TFIDF_OUTPUT_FILE = "data/news_with_tfidf.json"
CONFUSION_MATRIX_DIR = "data"
RESULTS_FILE = "data/model_evaluation_results.txt"
VISUALIZATION_DIR = "data/visualizations"

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# # Steps
# print("Starting preprocessing...")
# preprocess_news_articles(RAW_DATA_FILE, PREPROCESSED_DATA_FILE)

# print("Calculating sentiments...")
# calculate_sentiments(PREPROCESSED_DATA_FILE, POSITIVE_WORDS_FILE, NEGATIVE_WORDS_FILE, SENTIMENT_OUTPUT_FILE)

# print("Generating TF-IDF representation...")
# generate_tfidf(SENTIMENT_OUTPUT_FILE, TFIDF_CSV_FILE, TFIDF_OUTPUT_FILE)

# print("Evaluating models...")
# X, y, label_encoder = load_data(TFIDF_OUTPUT_FILE)
# results = train_and_evaluate_models(X, y)
# plot_confusion_matrices(results, label_encoder, CONFUSION_MATRIX_DIR)
# save_results(results, RESULTS_FILE)

print("Generating visualizations...")
generate_word_clouds(SENTIMENT_OUTPUT_FILE, VISUALIZATION_DIR)
generate_sentiment_distribution(SENTIMENT_OUTPUT_FILE, VISUALIZATION_DIR)


print("Pipeline complete! All outputs have been saved.")
