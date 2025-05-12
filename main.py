import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def create_features(data, target_days=1):
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()

    # Create lagged features
    for lag in [1, 5, 10, 20]:
        df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

    # Create rolling window features (using only past data)
    for window in [5, 10, 20]:
        df[f'Returns_rolling_mean_{window}'] = df['Returns'].rolling(window=window).mean().shift(1)
        df[f'Returns_rolling_std_{window}'] = df['Returns'].rolling(window=window).std().shift(1)

    # Create price-based features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    # Calculate future volatility (target variable)
    df['Future_Volatility'] = df['Returns'].rolling(window=target_days).std().shift(-target_days) * np.sqrt(252)

    return df.dropna()

def main():
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2025-04-25'
    target_days = 20  # Predict volatility 20 days ahead

    # Load data
    data = pd.read_csv(f"data_cache/{ticker}_data.csv", index_col=0, parse_dates=True)
    data = data.sort_index().loc[start_date:end_date]

    # Create features
    data = create_features(data, target_days)

    # Split data (maintaining temporal order)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    # Prepare features and target
    features = [col for col in data.columns if col not in ['Future_Volatility', 'Date']]
    X_train = train_data[features]
    y_train = train_data['Future_Volatility']
    X_test = test_data[features]
    y_test = test_data['Future_Volatility']

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Train Random Forest with best parameters
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train_scaled, y_train)

    # Make predictions
    train_pred = best_rf.predict(X_train_scaled)
    test_pred = best_rf.predict(X_test_scaled)

    # Calculate metrics
    print("Training Metrics:")
    print(calculate_metrics(y_train, train_pred))
    print("\nTest Metrics:")
    print(calculate_metrics(y_test, test_pred))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, y_train, label='Train Actual')
    plt.plot(test_data.index, y_test, label='Test Actual')
    plt.plot(test_data.index, test_pred, label='Test Predicted')
    plt.title(f'{ticker} Future Volatility Forecasting with Random Forest')
    plt.legend()
    plt.show()

    # Print feature importances
    importances = best_rf.feature_importances_
    feature_imp = pd.DataFrame(sorted(zip(importances, features)), columns=['Value','Feature'])
    print("\nTop 10 most important features:")
    print(feature_imp.nlargest(10, 'Value'))

if __name__ == "__main__":
    main()