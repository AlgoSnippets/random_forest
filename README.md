## Stock Volatility Prediction with Random Forest

This repository contains a Python script for predicting stock volatility using a Random Forest regression model. The script processes historical stock data, trains a machine learning model, evaluates its performance, and visualizes the results.

---

### **Features**

- Loads historical stock price data from CSV.
- Computes daily returns and rolling volatility.
- Scales features for robust model training.
- Prepares time series data for supervised learning.
- Trains a Random Forest regressor to predict future volatility.
- Evaluates model performance with standard regression metrics.
- Visualizes actual vs. predicted volatility.

---

### **Requirements**

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

### **Usage**

1. **Prepare Data**
   - Download historical stock data (e.g., from Yahoo Finance) and save as `data_cache/{TICKER}_data.csv`.
   - The CSV should include at least a `Date` column (as index) and a `Close` column.
2. **Edit Parameters**
   - In `main.py`, set the desired ticker, start date, and end date:

```python
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2025-04-25'
```

3. **Run the Script**

```bash
python main.py
```

The script will: - Print training and test set metrics (MSE, RMSE, MAE, R²). - Display a plot comparing actual and predicted volatility.

---

### **How It Works**

- **Feature Engineering:**
  - Calculates daily returns:

$$
\text{Returns}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}
$$

    - Calculates rolling volatility (annualized standard deviation over a 20-day window).

- **Data Preparation:**
  - Scales volatility using MinMaxScaler.
  - Creates input sequences of length 20 for supervised learning.
- **Model Training:**
  - Splits data into training (80%) and test (20%) sets.
  - Trains a Random Forest regressor on the training set.
- **Evaluation:**
  - Computes MSE, RMSE, MAE, and R² on both training and test sets.
  - Plots actual vs. predicted volatility for visual inspection.

---

### **Customization**

- Change the `window_size` parameter to adjust the lookback period.
- Tune Random Forest hyperparameters (e.g., `n_estimators`) for improved performance.
- Adapt the script for different tickers or data sources.

---

### **Example Output**

```
Training Metrics:
{'MSE': 0.0001, 'RMSE': 0.01, 'MAE': 0.008, 'R2': 0.95}

Test Metrics:
{'MSE': 0.0002, 'RMSE': 0.014, 'MAE': 0.011, 'R2': 0.90}
```

A plot will be shown comparing actual and predicted volatility over time.

---

### **License**

This project is released under the MIT License.

---

### **Acknowledgments**

- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

---
