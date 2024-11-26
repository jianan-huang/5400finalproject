import requests
import pandas as pd

# News API Key (replace with your key)
api_key = '328c27b7590e4abaac27d06a4ae2a8fd'

# List of popular tech companies to search for news
tech_companies = ["Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Nvidia", "Netflix", "Intel", "IBM"]

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
for company in tech_companies:
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

# Convert all articles to a DataFrame
df = pd.DataFrame(all_articles)

# Save to CSV file
df.to_csv('data/tech_financial_news_10_companies.csv', index=False)
print("Data saved to 'tech_financial_news_10_companies.csv'")














