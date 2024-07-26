# Scraper and Data Analysis Project

This repository contains a collection of scripts for web scraping and data analysis, with a focus on time series data and book information.

## Project Components

### 1. Web Scrapers
- `TimeSeriesScraper.py`: Scrapes time series data (likely financial data based on the results).
- `bookScraper.py`: Scrapes book information from Goodreads.

### 2. Data Analysis Scripts
- `advanced_analytics.py`: Performs advanced analytics on the scraped data.
- `data_checks.py`: Runs data quality checks and validations.
- `high_level_strategy.py`: Implements high-level trading or analysis strategies.
- `non_linear_model.py`: Applies non-linear modeling techniques to the data.

### 3. Results and Visualizations
- `TimeSeriesResults/`: Contains visualizations of time series analysis, including:
  - Trading volume analysis
  - Returns with anomalies
  - Volatility clustering
  - Autocorrelation of squared returns
- `bookscraper results/`: Includes results from the book scraping, such as:
  - JSON data of scraped books
  - Rating distribution visualization
- `high_level_strategy_results/`: Visualizations from the high-level strategy analysis:
  - Spread Z-Score
  - Correlation heat maps
  - Feature importance
  - Scatter representations

## Installation

To set up this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Jonathan-321/scraper.git
   ```
2. Navigate to the project directory:
   ```
   cd scraper
   ```
3. Install required dependencies (you may want to use a virtual environment):
   ```
   pip install -r requirements.txt
   ```

## Usage

Each script can be run independently. For example:

```
python TimeSeriesScraper.py
python bookScraper.py
python advanced_analytics.py
```

Ensure you have the necessary data files or API access for the scrapers to function correctly.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

MIT License
Copyright (c) 2024 
