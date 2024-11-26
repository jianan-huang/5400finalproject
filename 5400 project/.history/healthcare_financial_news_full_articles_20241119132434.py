from bs4 import BeautifulSoup
import requests
import pandas as pd

# Load the CSV file with articles and URLs
df = pd.read_csv('data/healthcare_financial_news_10_companies.csv')

# Function to get the full content of an article
def get_full_article(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            full_text = ' '.join([para.get_text() for para in paragraphs])
            return full_text
        else:
            return 'Error retrieving article'
    except Exception as e:
        return f'Error: {e}'

# Add a new column for full article content
df['full_content'] = df['url'].apply(get_full_article)

# Save the updated dataframe to a new CSV
df.to_csv('data/healthcare_financial_news_full_articles.csv', index=False)
print("Data saved with full articles to 'healthcare_financial_news_full_articles.csv'")