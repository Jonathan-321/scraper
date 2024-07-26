import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import random
import re

class GoodreadsScraper:
    def __init__(self, base_url="https://www.goodreads.com"):
        self.base_url = base_url
        self.books = []

    def fetch_page(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        return BeautifulSoup(response.text, 'html.parser')

    def scrape_book(self, book_element):
        try:
            title_element = book_element.find('a', class_='bookTitle')
            author_element = book_element.find('a', class_='authorName')
            rating_element = book_element.find('span', class_='greyText smallText')
            
            title = title_element.text.strip() if title_element else "N/A"
            author = author_element.text.strip() if author_element else "N/A"
            
            rating = None
            if rating_element:
                rating_text = rating_element.text.strip()
                rating_match = re.search(r'avg rating (\d+\.\d+)', rating_text)
                if rating_match:
                    rating = float(rating_match.group(1))
            
            if title == "N/A" and author == "N/A":
                return None  # Skip entries where both title and author are N/A
            
            print(f"Scraped: {title} by {author} - Rating: {rating}")  # Debugging line
            return {'title': title, 'author': author, 'rating': rating}
        except Exception as e:
            print(f"Error scraping book: {e}")
            return None

    def scrape_catalog(self, genre="fiction", pages=3):
        for page in range(1, pages + 1):
            print(f"Scraping page {page}...")
            url = f"{self.base_url}/shelf/show/{genre}?page={page}"
            soup = self.fetch_page(url)
            
            book_elements = soup.find_all('div', class_='elementList')
            print(f"Number of book elements found on page {page}: {len(book_elements)}")
            
            for book_element in book_elements:
                book_data = self.scrape_book(book_element)
                if book_data:
                    self.books.append(book_data)
            
            sleep(random.uniform(2, 5))  # Be nice to Goodreads servers

        # Remove duplicates
        self.books = [dict(t) for t in {tuple(d.items()) for d in self.books}]
        print(f"Total unique books scraped: {len(self.books)}")

    def save_to_json(self, filename='goodreads_books4.json'):
        with open(filename, 'w') as f:
            json.dump(self.books, f, indent=2)
        print(f"Data saved to {filename}")

    def analyze_data(self):
        if not self.books:
            print("No data to analyze.")
            return

        df = pd.DataFrame(self.books)
        
        # Basic statistics
        print(df.describe())
        
        # Top 5 highest rated books
        print("\nTop 5 highest rated books:")
        print(df.sort_values('rating', ascending=False).head())
        
        # Rating distribution
        plt.figure(figsize=(10, 6))
        df['rating'].hist(bins=20)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.savefig('goodreads_rating_distribution4.png')
        plt.close()

        print("Analysis complete. Check the generated PNG file for visualization.")

        # Additional analysis
        print("\nNumber of books with ratings:", df['rating'].notna().sum())
        print("Number of books without ratings:", df['rating'].isna().sum())
        print("\nTop 5 authors by number of books:")
        print(df['author'].value_counts().head())
if __name__ == "__main__":
    scraper = GoodreadsScraper()
    scraper.scrape_catalog(genre="fiction", pages=3)
    scraper.save_to_json()
    scraper.analyze_data()