import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
import datetime
import requests
from bs4 import BeautifulSoup
import os

class DataExtractor:
    """
    A class to fetch stock prices and news articles, and save them to the data folder.
    """
    def __init__(self, company_name, symbol, start_date, end_date, news_api_key):
        self.company_name = company_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.news_api_key = news_api_key
        self.newsapi = NewsApiClient(api_key=news_api_key)
        os.makedirs("data", exist_ok=True)

    def fetch_stock_data(self):
        """
        Fetch historical stock data using yfinance and save to a JSON file.
        """
        stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        stock_data = stock_data.reset_index()
        stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        stock_data.to_json("data/stock_prices.json", orient="records", date_format="iso")
        print("Stock prices saved to 'data/stock_prices.json'.")

    def fetch_news(self):
        """
        Fetch news articles from NewsAPI for the specified company and time range.
        """
        all_articles = []
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        delta = datetime.timedelta(days=7)

        while start <= end:
            from_date = start.strftime("%Y-%m-%d")
            to_date = (start + delta).strftime("%Y-%m-%d")

            articles = self.newsapi.get_everything(
                q=self.company_name,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="relevancy",
                page=1,
                page_size=100
            )
            if articles['status'] == "ok":
                all_articles.extend(articles['articles'])
            start += delta

        return all_articles

    def fetch_full_content(self, url):
        """
        Fetch the full content of a news article from its URL.

        Parameters:
        url (str): URL of the news article.

        Returns:
        str: Full article content as text.
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all('p')
            content = " ".join([para.get_text() for para in paragraphs])
            return content.strip()
        except Exception as e:
            print(f"Failed to fetch content from {url}: {e}")
            return None

    def save_news_articles(self):
        """
        Fetch news articles, including full content, and save to a JSON file.
        """
        news_articles = self.fetch_news()
        news_data = []
        for article in news_articles:
            full_content = self.fetch_full_content(article['url'])
            news_data.append({
                "source": article['source']['name'],
                "author": article['author'],
                "title": article['title'],
                "description": article['description'],
                "url": article['url'],
                "published_at": article['publishedAt'],
                "content": full_content or article.get('content', "")
            })

        news_df = pd.DataFrame(news_data)
        news_df.to_json("data/news_articles_with_full_content.json", orient="records", date_format="iso")
        print("News articles saved to 'data/news_articles_with_full_content.json'.")
