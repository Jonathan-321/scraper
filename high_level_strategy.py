import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import warnings
from textblob import TextBlob
import py_vollib.black_scholes.implied_volatility as iv
from scipy.stats import norm
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

def visualize_multi_factor_model(model, X, y):
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, factor in enumerate(X.columns):
        sns.scatterplot(x=X[factor], y=y, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{factor} vs Future Return')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    corr = pd.concat([X, y], axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Factors')
    plt.show()

    # Feature importance
    importance = pd.DataFrame({'factor': X.columns, 'importance': np.abs(model.coef_)})
    importance = importance.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='factor', data=importance)
    plt.title('Feature Importance')
    plt.show() 

def create_multi_factor_model(df):
    # Create factors
    df['Market_Return'] = df['Close'].pct_change()
    df['Size'] = np.log(df['Volume'] * df['Close'])
    df['Momentum'] = df['Close'].pct_change(20)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # Target variable: 5-day future return
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    
    # Prepare data for regression, dropping all NaN values
    features = ['Market_Return', 'Size', 'Momentum', 'Volatility', 'Future_Return']
    data = df[features].dropna()
    
    X = data[['Market_Return', 'Size', 'Momentum', 'Volatility']]
    y = data['Future_Return']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print results
    for factor, coef in zip(X.columns, model.coef_):
        print(f"{factor}: {coef}")
    print(f"R-squared: {model.score(X_test, y_test)}")

    return model, X, y   

def detect_market_regimes(df, n_regimes=2):
    returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=42)
    
    try:
        model.fit(returns)
    except Exception as e:
        print(f"Warning: HMM model did not converge. Error: {str(e)}")
        return None
    
    hidden_states = model.predict(returns)
    
    for i in range(n_regimes):
        mask = hidden_states == i
        print(f"Regime {i}:")
        print(f"  Mean return: {returns[mask].mean():.4f}")
        print(f"  Volatility: {returns[mask].std():.4f}")
        print(f"  Frequency: {mask.sum() / len(mask):.4f}")

    return hidden_states


def visualize_market_regimes(df, hidden_states):
    # Time series plot with regime overlay
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'])
    plt.scatter(df.index, df['Close'], c=hidden_states, cmap='viridis', alpha=0.5)
    plt.title('Stock Price with Market Regimes')
    plt.colorbar(ticks=range(len(np.unique(hidden_states))))
    plt.show()

    # Histogram of returns for each regime
    returns = df['Close'].pct_change().dropna()
    plt.figure(figsize=(12, 6))
    for i in np.unique(hidden_states):
        sns.histplot(returns[hidden_states == i], kde=True, label=f'Regime {i}')
    plt.title('Distribution of Returns by Regime')
    plt.legend()
    plt.show()


def analyze_pairs_trading(symbol1, symbol2, start_date, end_date):
    stock1 = yf.Ticker(symbol1).history(start=start_date, end=end_date)['Close']
    stock2 = yf.Ticker(symbol2).history(start=start_date, end=end_date)['Close']
    
    # Calculate spread
    spread = np.log(stock1) - np.log(stock2)
    
    # Test for cointegration
    score, pvalue, _ = coint(stock1, stock2)
    print(f"Cointegration p-value: {pvalue:.4f}")
    
    # Calculate z-score of spread
    z_score = (spread - spread.mean()) / spread.std()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(z_score)
    plt.axhline(0, color='black')
    plt.axhline(1.5, color='red', linestyle='--')
    plt.axhline(-1.5, color='green', linestyle='--')
    plt.title(f"{symbol1} - {symbol2} Spread Z-Score")
    plt.show()


def visualize_pairs_trading(symbol1, symbol2, start_date, end_date):
    stock1 = yf.Ticker(symbol1).history(start=start_date, end=end_date)['Close']
    stock2 = yf.Ticker(symbol2).history(start=start_date, end=end_date)['Close']

    # Price time series
    plt.figure(figsize=(15, 7))
    plt.plot(stock1.index, stock1 / stock1.iloc[0], label=symbol1)
    plt.plot(stock2.index, stock2 / stock2.iloc[0], label=symbol2)
    plt.title(f'Normalized Price Comparison: {symbol1} vs {symbol2}')
    plt.legend()
    plt.show()

    # Rolling correlation
    rolling_corr = stock1.rolling(window=30).corr(stock2)
    plt.figure(figsize=(15, 7))
    plt.plot(rolling_corr.index, rolling_corr)
    plt.title(f'30-Day Rolling Correlation: {symbol1} vs {symbol2}')
    plt.show()    

def analyze_news_sentiment(symbol, start_date, end_date):
    # Fetch news
    ticker = yf.Ticker(symbol)
    news = ticker.news
    
    # Analyze sentiment
    sentiments = []
    for article in news:
        blob = TextBlob(article['title'])
        sentiments.append(blob.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiments)
    print(f"Average sentiment: {avg_sentiment}")
    
    # Fetch stock data
    stock_data = ticker.history(start=start_date, end=end_date)
    
    # Compare sentiment with stock performance
    correlation = np.corrcoef(sentiments, stock_data['Close'][-len(sentiments):])[0, 1]
    print(f"Correlation between sentiment and stock price: {correlation}")


def visualize_news_sentiment(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    news = ticker.news
    stock_data = ticker.history(start=start_date, end=end_date)

    sentiments = [TextBlob(article['title']).sentiment.polarity for article in news]
    sentiment_dates = [pd.to_datetime(article['providerPublishTime'], unit='s') for article in news]

    plt.figure(figsize=(15, 7))
    plt.plot(stock_data.index, stock_data['Close'], label='Stock Price')
    plt.scatter(sentiment_dates, [stock_data['Close'].iloc[stock_data.index.get_loc(date, method='nearest')] for date in sentiment_dates], 
                c=sentiments, cmap='RdYlGn', label='Sentiment')
    plt.colorbar(label='Sentiment Polarity')
    plt.title(f'{symbol} Stock Price and News Sentiment')
    plt.legend()
    plt.show()

def calculate_implied_distribution(symbol, expiration_date):
    # Fetch options data
    options = yf.Ticker(symbol).options
    option_chain = yf.Ticker(symbol).option_chain(expiration_date)
    
    # Calculate implied volatility
    ivs = [iv.calculate(implied_volatility, option['lastPrice'], option['strike'], option['expiration'], option['underlyingPrice'], option['type']) for _, option in option_chain.iterrows()]
    
    # Calculate implied probability distribution
    ivs = np.array(ivs)
    std_dev = ivs.mean()
    mean = 0
    x = np.linspace(-3*std_dev, 3*std_dev, 100)
    y = norm.pdf(x, mean, std_dev)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.title(f"{symbol} Implied Probability Distribution")
    plt.show()
    

if __name__ == "__main__":
    # Load data
    symbol = "AAPL"
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)

    # Multi-Factor Model
    print("Multi-Factor Model Results:")
    model, X, y = create_multi_factor_model(df)
    visualize_multi_factor_model(model, X, y)

    # Market Regime Detection
    print("\nMarket Regime Detection:")
    hidden_states = detect_market_regimes(df)
    if hidden_states is not None:
        visualize_market_regimes(df, hidden_states)

    # Pairs Trading Analysis
    print("\nPairs Trading Analysis (AAPL vs MSFT):")
    analyze_pairs_trading("AAPL", "MSFT", start_date, end_date)
    visualize_pairs_trading("AAPL", "MSFT", start_date, end_date)

    # News Sentiment Analysis
    print("\nNews Sentiment Analysis:")
    analyze_news_sentiment(symbol, start_date, end_date)
    visualize_news_sentiment(symbol, start_date, end_date)

    # Options-Implied Probability Distribution
    print("\nOptions-Implied Probability Distribution:")
    available_dates = yf.Ticker(symbol).options
    if available_dates:
        next_available_date = min(date for date in available_dates if pd.to_datetime(date) > pd.Timestamp.now())
        calculate_implied_distribution(symbol, next_available_date)
    else:
        print("No option dates available.")

    # ... (keep the investment strategy implications)
""" ```

This enhanced script addresses the following:

1. Multi-Factor Model: Added visualizations for scatter plots, correlation heatmap, and feature importance.
2. Market Regime Detection: Added time series plot with regime overlay and histogram of returns by regime.
3. Pairs Trading: Added normalized price comparison and rolling correlation plots.
4. News Sentiment: Added a visualization that combines stock price and sentiment.
5. Options-Implied Distribution: Fixed the error by selecting the next available option date.

To use this script:

1. Replace your current `high_level_strategy.py` with this updated version.
2. Install additional required libraries: `pip install seaborn`
3. Run the script: `python high_level_strategy.py`

This script will generate multiple visualizations for each analysis, providing a more comprehensive view of the data and results. The visualizations will help in interpreting the results more intuitively and identifying patterns or relationships that might not be apparent from the numerical output alone.

Remember to interpret the results cautiously:

1. The multi-factor model still shows a negative R-squared, indicating poor fit. Consider exploring non-linear models or additional factors.
2. The market regime detection is not converging reliably. You might need to adjust the model parameters or consider alternative approaches.
3. The pairs trading analysis shows a high p-value, suggesting AAPL and MSFT might not be good candidates for this strategy.
4. The news sentiment analysis shows a weak correlation with stock price. Consider analyzing the content more deeply or using more sophisticated sentiment analysis techniques.

These visualizations and analyses provide a starting point for further investigation. As you refine your approach, you may want to consider more advanced techniques or incorporate additional data sources to improve the reliability and predictive power of your models.
""" 