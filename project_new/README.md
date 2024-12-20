# Financial News Sentiment Analysis and Stock Price Prediction

## Project Overview
This project aims to analyze the relationship between financial news sentiment and stock price fluctuations. It involves extracting stock prices and financial news articles, processing the data, performing sentiment analysis, and applying machine learning models to predict sentiment and understand its correlation with stock prices.

## Features

### 1. Data Extraction
- **News Articles:** Fetch news articles using the **NewsAPI**.
- **Stock Prices:** Retrieve stock price data using the **Yahoo Finance API**.

### 2. Text Preprocessing
- **Tokenization and Cleaning:** Process raw text data to remove special characters, numbers, and unwanted symbols.
- **Stopword Removal:** Eliminate common stopwords to retain meaningful words.
- **Stemming:** Reduce words to their root forms to unify vocabulary.

### 3. Sentiment Analysis
- **Sentiment Scores:** Compute sentiment scores for news articles using predefined positive and negative word dictionaries.
- this score is used as Further training labels

### 4. TF-IDF Representation
- Represent cleaned text data using **Term Frequency-Inverse Document Frequency (TF-IDF)** to quantify the importance of words in the corpus.

### 5. Machine Learning Models
- Train classifiers to predict sentiment:
  - **Support Vector Machine (SVM)**
  - **Random Forest**
  - **Naive Bayes**
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - ROC-AUC

### 6. Visualization
Generate insightful visualizations to explore sentiment and its impact on stock prices:
- **Word Clouds:** For positive and negative sentiment words.
- **Sentiment Distribution:** Bar and pie charts to show the distribution of sentiment scores.
- **Time-Series Comparison:** Compare sentiment trends with stock price fluctuations over time.

## Workflow Summary
1. **Data Collection:** Fetch stock prices and news articles via APIs.
2. **Preprocessing:** Clean and tokenize the text data.
3. **Sentiment Analysis:** Calculate sentiment scores using word dictionaries.
4. **Feature Extraction:** Convert text data into TF-IDF vectors.
5. **Model Training:** Train machine learning models to classify sentiment.
6. **Evaluation:** Evaluate models and analyze performance metrics.
7. **Visualization:** Create visualizations to interpret results and trends.

## Goals
- Understand the impact of financial news sentiment on stock prices.
- Predict sentiment scores for unseen financial news.
- Identify trends and patterns between news sentiment and stock price movements.

---

**Tools & Libraries:**
- Python
- NewsAPI, Yahoo Finance API
- NLTK, Scikit-learn, Matplotlib, Seaborn
- WordCloud, Pandas, NumPy
