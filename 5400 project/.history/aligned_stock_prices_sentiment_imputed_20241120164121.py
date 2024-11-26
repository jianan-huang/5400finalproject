import pandas as pd

# Load the aligned dataset
aligned_df = pd.read_csv('data/aligned_stock_prices_sentiment.csv')

# Convert the 'date' column to datetime format to work with time series data
aligned_df['date'] = pd.to_datetime(aligned_df['date'])

# Sort values by company and date to apply time series imputation correctly
aligned_df.sort_values(by=['company', 'date'], inplace=True)

# Handling Missing Sentiment Scores
# 1. Apply forward fill to fill missing sentiment scores based on time series
aligned_df['daily_sentiment_score'] = aligned_df.groupby('company')['daily_sentiment_score'].ffill()

# 2. Apply backward fill to cover any remaining gaps
aligned_df['daily_sentiment_score'] = aligned_df.groupby('company')['daily_sentiment_score'].bfill()

# 3. For any remaining NaN values (in case the first/last values are missing),
# fill them with the average sentiment score across similar companies (by industry)
# Assuming a mapping between companies and industries:
industry_mapping = {
    "Apple": "Technology", "Microsoft": "Technology", "Google": "Technology", "Amazon": "Technology", "Tesla": "Technology",
    "Meta": "Technology", "Nvidia": "Technology", "Netflix": "Technology", "Intel": "Technology", "IBM": "Technology",
    "Pfizer": "Healthcare", "Moderna": "Healthcare", "Johnson & Johnson": "Healthcare", "AbbVie": "Healthcare",
    "Merck & Co.": "Healthcare", "Amgen": "Healthcare", "Gilead Sciences": "Healthcare", "Bristol-Myers Squibb": "Healthcare",
    "Eli Lilly and Company": "Healthcare", "Biogen": "Healthcare",
    "JPMorgan Chase": "Finance", "Goldman Sachs": "Finance", "Bank of America": "Finance", "Morgan Stanley": "Finance",
    "Citigroup": "Finance", "Wells Fargo": "Finance", "American Express": "Finance", "BlackRock": "Finance",
    "Charles Schwab": "Finance", "Mastercard": "Finance"
}

# Map each company to its industry
aligned_df['industry'] = aligned_df['company'].map(industry_mapping)

# Fill remaining NaN values with the average sentiment score across companies within the same industry
aligned_df['daily_sentiment_score'] = aligned_df.groupby('industry')['daily_sentiment_score'].transform(lambda x: x.fillna(x.mean()))

# Drop the 'industry' column as it's no longer needed
aligned_df.drop(columns=['industry'], inplace=True)

# Save the updated dataset with imputed sentiment scores
aligned_df.to_csv('data/aligned_stock_prices_sentiment_imputed.csv', index=False)

print("Missing sentiment scores have been handled and the updated dataset has been saved.")
