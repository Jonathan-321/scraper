import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

def prepare_data(symbol, start_date, end_date):
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    
    df['Market_Return'] = df['Close'].pct_change()
    df['Size'] = np.log(df['Volume'] * df['Close'])
    df['Momentum'] = df['Close'].pct_change(20)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['Future_Return'] = df['Close'].pct_change(5).shift(-5)
    
    df = df.dropna()
    
    X = df[['Market_Return', 'Size', 'Momentum', 'Volatility']]
    y = df['Future_Return']
    
    return X, y

def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    return r2, mse, cv_scores.mean()

# Prepare data
symbol = "AAPL"
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=5)  # Using 5 years of data for more robust results
X, y = prepare_data(symbol, start_date, end_date)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf'),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
}

# Evaluate models
results = {}
for name, model in models.items():
    r2, mse, cv_score = evaluate_model(model, X, y)
    results[name] = {"R-squared": r2, "MSE": mse, "Cross-validation score": cv_score}

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

# Feature importance for the best performing model (assuming it's Random Forest)
best_model = models["Random Forest"]
best_model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)