import pytest
import os
import sys
import pandas as pd

# Dynamically add the src directory to Python paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))



from src.sentiment_analysis.sentiment import load_sentiment_words, calculate_sentiment, calculate_sentiments

# File paths for test data
POS_FILE = "tests/test_data/test_positive_words.txt"
NEG_FILE = "tests/test_data/test_negative_words.txt"
INPUT_FILE = "tests/test_data/test_preprocessed_news.json"
OUTPUT_FILE = "tests/test_data/test_output.json"

# Create functions for test sentiments

def test_load_sentiment_words():
    """Test loading positive and negative words."""
    positive_words, negative_words = load_sentiment_words(POS_FILE, NEG_FILE)
    assert "good" in positive_words
    assert "bad" in negative_words
    assert len(positive_words) == 5
    assert len(negative_words) == 5


@pytest.mark.parametrize(
    "text, expected_score, expected_polarity",
    [
        ("the product is great and excellent", 2, "positive"),
        ("the service was terrible and bad", -2, "negative"),
        ("it was a good day but a bit sad", 0, "positive"),
        ("", 0, "positive"),  # Empty content
    ],
)
def test_calculate_sentiment(text, expected_score, expected_polarity):
    """Test sentiment score and polarity calculation."""
    positive_words, negative_words = load_sentiment_words(POS_FILE, NEG_FILE)
    score, polarity = calculate_sentiment(text, positive_words, negative_words)
    assert score == expected_score
    assert polarity == expected_polarity


def test_calculate_sentiments():
    """Test the overall pipeline of sentiment calculation."""
    # Run the sentiment calculation
    calculate_sentiments(INPUT_FILE, POS_FILE, NEG_FILE, OUTPUT_FILE)

    # Load output data
    assert os.path.exists(OUTPUT_FILE), "Output file was not created."
    output_data = pd.read_json(OUTPUT_FILE)

    # Verify output
    assert "sentiment_score" in output_data.columns
    assert "polarity" in output_data.columns

    # Check specific results
    row1 = output_data.loc[output_data["id"] == 1]
    assert row1["sentiment_score"].iloc[0] == 2
    assert row1["polarity"].iloc[0] == "positive"

    row2 = output_data.loc[output_data["id"] == 2]
    assert row2["sentiment_score"].iloc[0] == -2
    assert row2["polarity"].iloc[0] == "negative"


if __name__ == "__main__":
    pytest.main()
