# Stock Volatility Prediction with Random Forest

This repository provides a Python implementation for predicting future stock volatility using a Random Forest regression model. The code uses historical price and volume data, engineers a variety of time-series features, and forecasts volatility over a user-defined horizon. The approach is suitable for financial analysts, data scientists, and machine learning practitioners interested in time-series forecasting and quantitative finance.

---

## Features

- **Time-Series Feature Engineering:**
  Lagged returns, rolling statistics, price ratios, and more.
- **Target Variable:**
  Predicts future realized volatility over a configurable window (e.g., 20 trading days).
- **Random Forest Regression:**
  Robust ensemble learning with hyperparameter tuning via grid search.
- **Evaluation \& Visualization:**
  Comprehensive metrics (MSE, RMSE, MAE, R²) and plots of actual vs. predicted volatility.
- **Feature Importance:**
  Ranks the top drivers of volatility predictions.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Usage

1. **Prepare Data**
   - Download historical stock data (e.g., from Yahoo Finance) and save as `data_cache/{TICKER}_data.csv`.
   - The CSV should contain columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
2. **Configure Parameters**
   - Edit `main.py` to set your ticker, date range, and prediction horizon:

```python
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2025-04-25'
target_days = 20  # Predict volatility 20 days ahead
```

3. **Run the Script**

```bash
python main.py
```

    - The script will print training and test metrics, plot actual vs. predicted volatility, and display the top 10 most important features.

---

## How It Works

- **Feature Engineering:**
  The script creates lagged and rolling features from price and volume, and computes ratios such as high/low and close/open.
- **Target Construction:**
  Future volatility is calculated as the rolling standard deviation of returns, annualized for comparability.
- **Model Training:**
  Data is split chronologically into training and test sets. Features are scaled, and Random Forest hyperparameters are optimized using grid search.
- **Evaluation:**
  Model performance is assessed with standard regression metrics and visualized for easy interpretation.
- **Feature Importance:**
  The script outputs the top features influencing volatility predictions.

---

## Example Output

```
Training Metrics:
{'MSE': 0.0001, 'RMSE': 0.01, 'MAE': 0.008, 'R2': 0.95}

Test Metrics:
{'MSE': 0.0002, 'RMSE': 0.014, 'MAE': 0.011, 'R2': 0.90}

Top 10 most important features:
      Value                Feature
0  0.1623      Returns_lag_1
1  0.1208      Returns_rolling_std_20
...
```

A plot will be shown comparing actual and predicted future volatility.

---

**Note:** This code is for educational and research purposes only. Not investment advice.
