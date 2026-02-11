import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

# Define parameters
tickers = ["SPY", "KO", "MA"]
train_end_date = "2024-02-29"
predict_start_date = "2024-03-01"
predict_end_date = "2024-03-31"
window_size = 60

# Create an Excel writer
writer = pd.ExcelWriter("stock_data_SPY.xlsx", engine="xlsxwriter")


def prepare_data(data):
    dataset = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled, scaler


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=(input_shape, 1)),
            LSTM(units=128, return_sequences=True),
            LSTM(units=64, return_sequences=False),
            Dense(units=25),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Fetch and predict data for each ticker
for ticker in tickers:
    try:
        data = yf.download(
            ticker, start="2021-01-01", end=train_end_date, progress=False
        )
        if data.empty:
            print(f"No data found for {ticker} in the given period.")
            continue

        dataset_scaled, scaler = prepare_data(data)
        X_train, y_train = create_sequences(dataset_scaled, window_size)
        if len(X_train) == 0:
            continue

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        model = build_lstm_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        predictions = []
        last_sequence = dataset_scaled[-window_size:].reshape((1, window_size, 1))

        # Predict day by day in March, updating the training set as we go
        trading_days = yf.download(
            "SPY", start=predict_start_date, end=predict_end_date, progress=False
        ).index
        for trading_day in trading_days:
            predicted_scaled = model.predict(last_sequence, verbose=0)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0, 0]
            predictions.append(
                {
                    "Date": trading_day.strftime("%Y-%m-%d"),
                    "Predicted_Price": predicted_price,
                }
            )

            new_entry = scaler.transform([[predicted_price]])[0, 0]
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = new_entry

        # Fetch actual prices for March
        actual_prices = yf.download(
            ticker, start=predict_start_date, end=predict_end_date, progress=False
        )["Close"]
        df_predictions = pd.DataFrame(predictions)
        df_predictions["Actual_Price"] = df_predictions["Date"].map(
            lambda d: actual_prices.get(d, None)
        )

        # Save to Excel
        df_predictions.to_excel(writer, sheet_name=ticker, index=False)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Save the Excel file
writer.close()
print("Stock data with predictions saved to stock_data.xlsx")
