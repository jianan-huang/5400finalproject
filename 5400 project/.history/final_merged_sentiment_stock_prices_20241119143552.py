import pandas as pd
financial_news_path = 'data/financial_news_with_vader_sentiment.csv'
financial_news_df = pd.read_csv(financial_news_path)
print(financial_news_df[['company', 'date', 'vader_sentiment_score']].isna().sum())
print(sentiment_aggregated.head())
print(sentiment_aggregated.isna().sum())


