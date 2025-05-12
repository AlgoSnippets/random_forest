import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def prepare_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].flatten())
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def main():
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2025-04-25'

    # Load and prepare data
    data = pd.read_csv(f"data_cache/{ticker}_data.csv", index_col=0, parse_dates=True)
    data = data.sort_index().loc[start_date:end_date]
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
    data = data.dropna()

    # Normalize volatility
    scaler = MinMaxScaler()
    scaled_volatility = scaler.fit_transform(data[['Volatility']])

    # Create sequences
    window_size = 20
    X, y = prepare_sequences(scaled_volatility, window_size)

    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Inverse scale predictions and actual values
    train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

    print("Training Metrics:")
    print(calculate_metrics(y_train, train_pred))
    print("\nTest Metrics:")
    print(calculate_metrics(y_test, test_pred))

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[window_size:split+window_size], y_train, label='Train Actual')
    plt.plot(data.index[split+window_size:], y_test, label='Test Actual')
    plt.plot(data.index[split+window_size:], test_pred, label='Test Predicted')
    plt.title(f'{ticker} Volatility Forecasting with Random Forest')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
