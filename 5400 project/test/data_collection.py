import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

def fetch_news(api_key, start_date, end_date, output_csv_path):
    # List of popular companies to search for news
    finance_companies = [
        "JPMorgan Chase", "Goldman Sachs", "Bank of America", "Morgan Stanley", "Citigroup", 
        "Wells Fargo", "American Express", "BlackRock", "Charles Schwab", "Mastercard"
    ]
    healthcare_companies = [
        "Pfizer", "Moderna", "Johnson & Johnson", "AbbVie", "Merck & Co.", "Amgen", "Gilead Sciences", "Bristol-Myers Squibb", "Eli Lilly and Company", "Biogen"
    ]
    tech_companies = [
        "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Nvidia", "Netflix", "Intel", "IBM"
    ]

    all_companies = finance_companies + healthcare_companies + tech_companies

    # Base URL for News API
    base_url = "https://newsapi.org/v2/everything"

    # Parameters for news crawling
    params = {
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,  # Maximum number of articles per request
        'from': start_date,
        'to': end_date
    }

    all_articles = []

    # Loop over each company to get articles
    for company in all_companies:
        params['q'] = company  # Search for each company name
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Collect articles in a structured format
            for article in articles:
                url = article['url']
                try:
                    full_content = scrape_full_content(url)
                    content = full_content if full_content else article.get('content', 'N/A')
                except Exception as e:
                    print(f"Error scraping content from {url}: {e}")
                    content = article.get('content', 'N/A')
                
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
        else:
            print(f"Error fetching data for {company}: {response.status_code} - {response.text}")
            if response.status_code == 426:
                print("Please adjust the date range to comply with your API plan limits.")
                return

    # Convert all articles to a DataFrame
    df = pd.DataFrame(all_articles)

    # Drop rows with no article content
    df = df[df['content'] != 'N/A']

    # Save to CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to '{output_csv_path}'")

def scrape_full_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Attempt to find the main content by selecting specific tags that often contain full article text
        article_body = soup.find('article')
        if article_body:
            paragraphs = article_body.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        full_content = ' '.join([p.get_text() for p in paragraphs if p.get_text().strip()])
        return full_content
    else:
        print(f"Failed to retrieve the page: {url}")
        return None

    # Adding delay to avoid getting blocked by the website
    time.sleep(random.uniform(1, 3))

# Example usage
if __name__ == "__main__":
    api_key = '328c27b7590e4abaac27d06a4ae2a8fd'  # Replace with your API key
    # start_date = input("Enter the start date (YYYY-MM-DD): ")
    # end_date = input("Enter the end date (YYYY-MM-DD): ")
    start_date =("2024-10-26")
    end_date =("2024-11-23")
    output_csv_path = 'Financial_news.csv'
    
    fetch_news(api_key, start_date, end_date, output_csv_path)

