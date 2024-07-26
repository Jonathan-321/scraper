import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def main():
    # List of stocks to collect data for
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Add more as needed
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    all_data = {}
    for ticker in stocks:
        print(f"Collecting data for {ticker}")
        all_data[ticker] = collect_stock_data(ticker, start_date, end_date)
    
    # Save to CSV files
    for ticker, data in all_data.items():
        data.to_csv(f"{ticker}_data.csv")
        print(f"Data for {ticker} saved to {ticker}_data.csv")

if __name__ == "__main__":
    main()