import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def calculate_technical_indicators(df):
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def perform_seasonal_decomposition(df):
    # Use a smaller period for decomposition
    decomposition = seasonal_decompose(df['Close'], model='additive', period=52)  # Using 52 weeks instead of 252 days
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.show()

def analyze_volatility_clustering(df):
    returns = df['Close'].pct_change().dropna()
    squared_returns = returns ** 2
    
    plt.figure(figsize=(12, 6))
    plt.plot(squared_returns.index, squared_returns)
    plt.title('Volatility Clustering')
    plt.xlabel('Date')
    plt.ylabel('Squared Returns')
    plt.show()
    
    # Autocorrelation of squared returns
    autocorr = [squared_returns.autocorr(lag=i) for i in range(1, 21)]
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 21), autocorr)
    plt.title('Autocorrelation of Squared Returns')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

def detect_anomalies(df):
    returns = df['Close'].pct_change().dropna()
    z_scores = stats.zscore(returns)
    threshold = 3
    anomalies = returns[abs(z_scores) > threshold]
    
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, returns, label='Returns')
    plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
    plt.title('Returns with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()

def analyze_intraday_patterns(symbol):
    # Fetch intraday data (1-minute intervals for the last 7 days)
    intraday_data = yf.download(symbol, period="5d", interval="1m")
    
    # Convert index to datetime and extract hour and minute
    intraday_data.index = pd.to_datetime(intraday_data.index)
    intraday_data['Hour'] = intraday_data.index.hour
    intraday_data['Minute'] = intraday_data.index.minute
    
    # Calculate average volume by hour
    hourly_volume = intraday_data.groupby('Hour')['Volume'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_volume.index, hourly_volume.values)
    plt.title('Average Intraday Volume by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Volume')
    plt.xticks(range(24))
    plt.show()

# Example usage
symbol = "AAPL"
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=1)

df = get_stock_data(symbol, start_date, end_date)
df = calculate_technical_indicators(df)

print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())

perform_seasonal_decomposition(df)
analyze_volatility_clustering(df)
detect_anomalies(df)
analyze_intraday_patterns(symbol)

# Save to CSV
df.to_csv(f"{symbol}_advanced_analysis.csv")