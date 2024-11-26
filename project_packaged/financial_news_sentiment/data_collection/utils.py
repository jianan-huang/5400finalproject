import requests
from bs4 import BeautifulSoup
import time
import random

def scrape_full_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
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

    time.sleep(random.uniform(1, 3))
