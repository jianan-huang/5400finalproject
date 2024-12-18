import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os


def generate_word_clouds(input_file, output_dir):
    """
    Generate word clouds for positive and negative sentiment text data.

    Args:
        input_file (str): Path to the input JSON file with 'filtered_content' and 'polarity'.
        output_dir (str): Directory to save the word cloud images.

    Returns:
        None
    """
    print("Generating word clouds...")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    news_data = pd.read_json(input_file)

    # Separate data by polarity
    positive_text = " ".join(news_data[news_data["polarity"] == "positive"]["filtered_content"].dropna())
    negative_text = " ".join(news_data[news_data["polarity"] == "negative"]["filtered_content"].dropna())

    # Generate and save word clouds
    for sentiment, text in [("positive", positive_text), ("negative", negative_text)]:
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud for {sentiment.capitalize()} Sentiment")
            output_path = os.path.join(output_dir, f"wordcloud_{sentiment}.png")
            plt.savefig(output_path)
            print(f"Word cloud for '{sentiment}' sentiment saved to '{output_path}'.")
            plt.close()
        else:
            print(f"No text data available for '{sentiment}' sentiment.")


def generate_sentiment_distribution(input_file, output_dir):
    """
    Generate sentiment distribution visualizations (bar chart and pie chart).

    Args:
        input_file (str): Path to the input JSON file with 'polarity'.
        output_dir (str): Directory to save the distribution plots.

    Returns:
        None
    """
    print("Generating sentiment distribution visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    news_data = pd.read_json(input_file)

    # Count sentiment distribution
    sentiment_counts = news_data["polarity"].value_counts()

    # Bar chart
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.title("Sentiment Distribution (Bar Chart)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    bar_chart_path = os.path.join(output_dir, "sentiment_distribution_bar.png")
    plt.savefig(bar_chart_path)
    print(f"Sentiment distribution bar chart saved to '{bar_chart_path}'.")
    plt.close()

    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140, colors=["#1f77b4", "#ff7f0e"])
    plt.title("Sentiment Distribution (Pie Chart)")
    pie_chart_path = os.path.join(output_dir, "sentiment_distribution_pie.png")
    plt.savefig(pie_chart_path)
    print(f"Sentiment distribution pie chart saved to '{pie_chart_path}'.")
    plt.close()
