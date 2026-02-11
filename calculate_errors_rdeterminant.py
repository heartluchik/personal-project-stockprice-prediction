import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the Excel file
file_path = "stock_data_from2023.xlsx"
xls = pd.ExcelFile(file_path)

# List of tickers (sheet names)
tickers = ["AAPL", "MSFT", "AMD", "NVDA"]


# Function to calculate performance metrics
def calculate_metrics(df):
    predicted = df["Predicted_Price"].values
    actual = df["Actual_Price"].values
    high = df["High"].values

    # Compute metrics
    rmse_actual = np.sqrt(mean_squared_error(actual, predicted))
    rmse_high = np.sqrt(mean_squared_error(high, predicted))
    r2_actual = r2_score(actual, predicted)
    r2_high = r2_score(high, predicted)
    mae_actual = np.mean(np.abs(actual - predicted))
    mae_high = np.mean(np.abs(high - predicted))
    mape_actual = np.mean(np.abs((actual - predicted) / actual)) * 100
    mape_high = np.mean(np.abs((high - predicted) / high)) * 100
    mse_actual = mean_squared_error(actual, predicted)
    mse_high = mean_squared_error(high, predicted)
    smape_actual = (
        np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
        * 100
    )
    smape_high = (
        np.mean(2 * np.abs(high - predicted) / (np.abs(high) + np.abs(predicted))) * 100
    )

    return {
        "RMSE (Predicted vs Actual)": rmse_actual,
        "RMSE (Predicted vs High)": rmse_high,
        "R² (Predicted vs Actual)": r2_actual,
        "R² (Predicted vs High)": r2_high,
        "MAE (Predicted vs Actual)": mae_actual,
        "MAE (Predicted vs High)": mae_high,
        "MAPE (Predicted vs Actual)": mape_actual,
        "MAPE (Predicted vs High)": mape_high,
        "MSE (Predicted vs Actual)": mse_actual,
        "MSE (Predicted vs High)": mse_high,
        "SMAPE (Predicted vs Actual)": smape_actual,
        "SMAPE (Predicted vs High)": smape_high,
    }


# Process each ticker sheet
for ticker in tickers:
    if ticker in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=ticker)
        metrics = calculate_metrics(df)

        # Save metrics to a text file
        with open(f"{ticker}_performance.txt", "w") as f:
            f.write(f"Performance Metrics for {ticker}:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

print("Performance metrics saved to text files.")
