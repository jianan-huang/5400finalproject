# Aggregate sentiment by date and company
sentiment_aggregated = news_df.groupby(['company', 'date']).agg({'vader_sentiment': 'first'}).reset_index()

# Load the merged stock and sentiment CSV
merged_csv_path = 'data/merged_sentiment_stock_prices_long_format.csv'
merged_df = pd.read_csv(merged_csv_path)

# Merge sentiment with stock price data
final_df = pd.merge(merged_df, sentiment_aggregated, on=['company', 'date'], how='left')

# Save the final merged dataset
final_df.to_csv('data/final_merged_sentiment_stock_prices.csv', index=False)
print("Final data saved to 'data/final_merged_sentiment_stock_prices.csv'")
