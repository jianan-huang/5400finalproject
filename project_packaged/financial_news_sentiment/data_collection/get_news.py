import requests
import pandas as pd
import os
import time
from financial_news_sentiment.data_collection.utils import scrape_full_content

def fetch_news(api_key, start_date, end_date, output_csv_path):
    companies = [
        "JPMorgan Chase", "Goldman Sachs", "Pfizer", "Moderna",
        "Apple", "Microsoft", "Tesla", "Nvidia"
    ]

    base_url = "https://newsapi.org/v2/everything"
    params = {
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,
        'from': start_date,
        'to': end_date
    }

    all_articles = []

    for company in companies:
        params['q'] = company
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            for article in articles:
                url = article['url']
                content = scrape_full_content(url) if url else article.get('content', 'N/A')

                all_articles.append({
                    'company': company,
                    'source': article['source']['name'],
                    'author': article.get('author', 'N/A'),
                    'title': article['title'],
                    'description': article.get('description', 'N/A'),
                    'url': url,
                    'published_at': article['publishedAt'],
                    'content': content
                })

            # Adding a delay to avoid getting rate-limited by the API
            time.sleep(1)

        else:
            print(f"Error fetching data for {company}: {response.status_code} - {response.text}")
            if response.status_code == 426:
                print("Please adjust the date range to comply with your API plan limits.")
                return

    df = pd.DataFrame(all_articles)
    df = df[df['content'] != 'N/A']

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to '{output_csv_path}'")

if __name__ == "__main__":
    # Define your API key, start date, end date, and output CSV path
    api_key = "328c27b7590e4abaac27d06a4ae2a8fd"  # Replace with your actual News API key
    start_date = "2024-10-27"
    end_date = "2024-11-24"
    output_csv_path = "data/raw/Financial_news.csv"

    # Call the fetch_news function
    fetch_news(api_key, start_date, end_date, output_csv_path)
