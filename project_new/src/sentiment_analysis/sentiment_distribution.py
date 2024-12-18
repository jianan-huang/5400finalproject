from collections import Counter
from .sentiment import SentimentAnalyzer

class SentimentDistribution:
    """
    Compute sentiment distribution in a dataset.
    """
    def __init__(self, data, text_column):
        self.data = data
        self.text_column = text_column

    def compute_distribution(self):
        """
        Compute and return sentiment counts.

        Returns:
        dict: Sentiment counts.
        """
        sentiments = [SentimentAnalyzer.analyze_sentiment(text) for text in self.data[self.text_column]]
        distribution = Counter(sentiments)
        print(f"Sentiment Distribution: {distribution}")
        return distribution
