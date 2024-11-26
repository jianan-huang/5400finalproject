import requests
import pandas as pd

# News API Key (replace with your key)
api_key = 'your_news_api_key'

# List of popular tech companies to search for news
finance_companies = ["JPMorgan Chase", "Goldman Sachs", "Bank of America", "Morgan Stanley", "Citigroup", "Wells Fargo", "American Express", "BlackRock", "Charles Schwab", "Mastercard"]

# Base URL for News API
base_url = "https://newsapi.org/v2/everything"

# Parameters for news crawling
params = {
    'apiKey': '328c27b7590e4abaac27d06a4ae2a8fd',
    'language': 'en',
    'sortBy': 'publishedAt',
    'pageSize': 100  # Maximum number of articles per request
}

all_articles = []

# Loop over each company to get articles
for company in finance_companies:
    params['q'] = company  # Search for each company name
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        
        # Collect articles in a structured format
        for article in articles:
            all_articles.append({
                'company': company,
                'source': article['source']['name'],
                'author': article.get('author', 'N/A'),
                'title': article['title'],
                'description': article.get('description', 'N/A'),
                'url': article['url'],
                'published_at': article['publishedAt'],
                'content': article.get('content', 'N/A')
            })
    else:
        print(f"Error fetching data for {company}: {response.status_code} - {response.text}")

# Ensure there are articles to work with before proceeding
if not all_articles:
    print("No articles fetched. Please check your API key or network connection.")
else:
    # Convert all articles to a DataFrame
    df = pd.DataFrame(all_articles)

    # Define keywords to filter articles related to stocks
    stock_keywords = ['stock', 'price', 'market', 'shares', 'trading', 'invest', 'financial', 'NASDAQ', 'NYSE', 'earnings']

    # Filter articles based on the presence of these keywords in the title, description, or content
    filtered_df = df[df['title'].str.contains('|'.join(stock_keywords), case=False, na=False) |
                     df['description'].str.contains('|'.join(stock_keywords), case=False, na=False) |
                     df['content'].str.contains('|'.join(stock_keywords), case=False, na=False)]

    # Save the filtered dataset to a CSV file
    filtered_df.to_csv('data/filtered_finance_financial_news.csv', index=False)
    print("Filtered data saved to 'filtered_finance_financial_news.csv'")