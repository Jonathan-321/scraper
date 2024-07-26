import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def load_csv_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def fetch_comparison_data(ticker, start_date, end_date):
    return yf.Ticker(ticker).history(start=start_date, end=end_date)

def check_data_completeness(df):
    # Check for missing dates
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    missing_dates = date_range.difference(df.index)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    return missing_dates, missing_values

def check_data_accuracy(df):
    # Check for negative prices
    negative_prices = df[['Open', 'High', 'Low', 'Close']] < 0
    
    # Check if Low <= Open, Close <= High
    price_order_violation = (df['Low'] > df['Open']) | (df['Low'] > df['Close']) | \
                            (df['High'] < df['Open']) | (df['High'] < df['Close'])
    
    return negative_prices.sum(), price_order_violation.sum()

def compare_with_other_source(df1, df2):
    # Align the dataframes
    df1, df2 = df1.align(df2, join='inner')
    
    # Calculate percentage difference
    pct_diff = ((df1 - df2) / df2).abs() * 100
    
    # Find significant differences (e.g., > 1%)
    significant_diff = pct_diff > 1
    
    return significant_diff.sum()


def check_missing_dates(missing_dates):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    business_days = pd.date_range(start=missing_dates.min(), end=missing_dates.max(), freq=us_bd)
    unexpected_missing = missing_dates.difference(business_days)
    return unexpected_missing

def main():
    # Load your data
    your_data = load_csv_data('AMZN_data.csv')
    
    # Fetch comparison data
    ticker = 'AMZN'  # Replace with your stock ticker
    comparison_data = fetch_comparison_data(ticker, your_data.index[0], your_data.index[-1])
    
    # Check data completeness
    missing_dates, missing_values = check_data_completeness(your_data)
    print(f"Missing dates: {len(missing_dates)}")
    print(f"Missing values:\n{missing_values}")
    
    # Check data accuracy
    negative_prices, price_order_violations = check_data_accuracy(your_data)
    print(f"Negative prices: {negative_prices.sum()}")
    print(f"Price order violations: {price_order_violations}")
    
    # Compare with other source
    significant_differences = compare_with_other_source(your_data, comparison_data)
    print(f"Significant differences with comparison source:\n{significant_differences}")
    
    # Additional checks
    print(f"Date range: {your_data.index.min()} to {your_data.index.max()}")
    print(f"Number of trading days: {len(your_data)}")
    print(f"Average daily volume: {your_data['Volume'].mean():.0f}")


    # In your main function, add:
    unexpected_missing = check_missing_dates(missing_dates)
    print(f"Unexpected missing dates: {len(unexpected_missing)}")
    if len(unexpected_missing) > 0:
        print(unexpected_missing)

if __name__ == "__main__":
    main()


""" ```

This script provides a comprehensive set of checks to validate your financial data:

1. Data Completeness:
   - Checks for missing dates in the dataset
   - Identifies any missing values in the columns

2. Data Accuracy:
   - Checks for negative prices (which shouldn't occur in stock data)
   - Verifies that the price order makes sense (Low <= Open, Close <= High)

3. Data Consistency:
   - Compares your data with another source (in this case, fetching fresh data from Yahoo Finance)
   - Identifies significant differences between the two sources

4. Additional Checks:
   - Verifies the date range of your data
   - Counts the number of trading days
   - Calculates average daily volume as a sanity check

To use this script:

1. Save your CSV data in the same directory as the script
2. Update the file name in the `load_csv_data()` function call
3. Set the correct ticker symbol in the `main()` function
4. Run the script

Interpreting the Results:

- Missing dates: There should be very few, if any, missing dates (except for weekends and holidays)
- Missing values: Ideally, there should be no missing values
- Negative prices: There should be no negative prices
- Price order violations: There should be no violations of the expected price order
- Significant differences: There should be very few differences when compared to a reliable source like Yahoo Finance

If you find discrepancies:

1. Cross-check with multiple sources (e.g., Yahoo Finance, Google Finance, your broker's data)
2. Look for known corporate actions (splits, dividends) that might explain differences
3. Check for data adjustments (some sources provide adjusted prices, others don't)
4. Consider contacting your data provider for clarification on any persistent discrepancies

Remember, small differences (< 1%) between sources are not uncommon due to differences in data collection times or methodologies. However, large or persistent differences should be investigated.

To further enhance trust in your data:

1. Implement these checks as part of your regular data update process
2. Keep logs of any data issues and how they were resolved
3. Periodically review and update your validation process
4. Consider using multiple data sources and implementing a voting system for the "correct" values

By rigorously validating your data, you're building a solid foundation for your trading strategy. This process will help you catch and correct issues early, preventing them from affecting your analysis and decision-making down the line. """