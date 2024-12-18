import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def generate_tfidf(input_file, tfidf_csv_file, output_file, max_features=1000):
    """
    Generate TF-IDF representation for the filtered content of a dataset.

    Parameters:
        input_file (str): Path to the input JSON file with 'filtered_content'.
        tfidf_csv_file (str): Path to save the TF-IDF matrix as a CSV file.
        output_file (str): Path to save the dataset with TF-IDF vectors.
        max_features (int): Maximum number of features for TF-IDF vectorization.

    Returns:
        None
    """
    print("Loading data...")
    news_data = pd.read_json(input_file)

    # Ensure filtered content exists
    if 'filtered_content' not in news_data.columns:
        raise KeyError("The input file must contain a 'filtered_content' column.")

    print("Extracting corpus from filtered content...")
    corpus = news_data['filtered_content'].dropna().tolist()

    # TF-IDF vectorization
    print("Generating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Convert TF-IDF matrix to DataFrame
    print("Creating TF-IDF DataFrame...")
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_features)

    # Save TF-IDF representation to CSV
    print(f"Saving TF-IDF representation to '{tfidf_csv_file}'...")
    tfidf_df.to_csv(tfidf_csv_file, index=False)

    # Add TF-IDF vectors to the original dataset
    print("Adding TF-IDF vectors to the dataset...")
    news_data['tfidf_vector'] = list(tfidf_matrix.toarray())

    # Save the updated dataset with TF-IDF vectors
    print(f"Saving updated dataset to '{output_file}'...")
    news_data.to_json(output_file, orient="records", date_format="iso")

    print("TF-IDF representation complete!")


if __name__ == "__main__":
    # File paths
    INPUT_FILE = "data/news_with_sentiments.json"
    TFIDF_CSV_FILE = "data/tfidf_representation.csv"
    OUTPUT_FILE = "data/news_with_tfidf.json"

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Generate TF-IDF representation
    generate_tfidf(INPUT_FILE, TFIDF_CSV_FILE, OUTPUT_FILE, max_features=1000)
