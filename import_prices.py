import yfinance as yf
import pandas as pd

# Define parameters
tickers = ["SPY", "KO", "MA"]
start_date = "2024-03-01"
end_date = "2024-03-30"

# Create an Excel writer
writer = pd.ExcelWriter("stock_data_historical_spy.xlsx", engine="xlsxwriter")

# Fetch and save data for each ticker
for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No data found for {ticker} in the given period.")
            continue
        data.to_excel(writer, sheet_name=ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Save the Excel file
writer.close()
print("Stock data saved to stock_data.xlsx")
