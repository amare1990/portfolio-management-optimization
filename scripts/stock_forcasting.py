"""Time-series forcasting using statistical and ML models."""

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the StockForecasting class
class StockForecasting:
    def __init__(self, processed_data, ticker):
        self.data = processed_data
        self.ticker = ticker



    def retrieve_data_by_ticker(self, ticker, inplace=False):
        """Will filter data by tickers and rename columns appropriately."""

        keep_columns = [f'Close {self.ticker}', f'High {self.ticker}', f'Low {self.ticker}', f'Open {self.ticker}', f'Volume {self.ticker}']

        if keep_columns:
            self.data = self.data[keep_columns]
            # If inplace is True, update self.data in place
            if inplace:
                self.data = self.data[keep_columns]
            else:
                # If inplace is False, return the filtered data
                return self.data[keep_columns]
        else:
            raise ValueError(f"{keep_columns} not found in data columns.")





    def split_data(self):
        """Split into train and test sets (80% train, 20% test)."""
        train_size = int(len(self.data) * 0.8)
        self.train_data, self.test_data = self.data[:train_size], self.data[train_size:]


        self.train_data = self.train_data[[f'Close {self.ticker}']]
        self.test_data = self.test_data[[f'Close {self.ticker}']]

        print("Train data columns:", self.train_data.columns)
        print("Test data columns:", self.test_data.columns)




    def arima_model(self):
        """
        Fit an ARIMA model on the stock data.
        """
        model = ARIMA(self.data[f'Close {self.ticker}'], order=(5, 1, 0))
        self.arima_model = model.fit()

    def forecast_arima(self):
        """
        Forecast future stock prices using the ARIMA model.
        """
        forecast = self.arima_model.predict(start=len(self.train_data), end=len(self.data)-1)
        # **Fix: Return forecast as a pandas Series with the correct column name**
        return pd.Series(forecast, index=self.test_data.index, name=f'Close {self.ticker}')


    def sarima_model(self):
        """
        Fit a SARIMA model on the stock data.
        """
        # Modify the seasonal_order and order to avoid conflict in AR lags
        model = SARIMAX(self.data[f'Close {self.ticker}'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))  # Example: Changed the seasonal period to 12
        self.sarima_model = model.fit()

    def forecast_sarima(self):
        """
        Forecast future stock prices using the SARIMA model.
        """
        forecast = self.sarima_model.forecast(steps=len(self.test_data))
        # **Fix: Return forecast as a pandas Series with the correct column name**
        return pd.Series(forecast, index=self.test_data.index, name=f'Close {self.ticker}')




    def build_lstm_model(self, look_back=60):
        """
        Build the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    def create_lstm_data(self, look_back):
        X, y = [], []
        for i in range(len(self.data) - look_back - 1):
            X.append(self.data.iloc[i:(i + look_back), 0].values)
            y.append(self.data.iloc[i + look_back, 0])
        return np.array(X), np.array(y)

    def train_lstm(self, look_back=60, epochs=5, batch_size=32):
        """
        Train the LSTM model.
        """
        X, y = self.create_lstm_data(look_back)

        # Split into training and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]


        model = self.build_lstm_model(look_back)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self.lstm_model = model
        self.X_test, self.y_test = X_test, y_test

    def forecast_lstm(self):
        """
        Forecast future stock prices using the trained LSTM model.
        """
        predicted_stock_price = self.lstm_model.predict(self.X_test)
        # Remove inverse_transform as data is already scaled in preprocessing
        # predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)
        # actual_stock_price = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        actual_stock_price = self.y_test.reshape(-1, 1)  # Reshape y_test for consistency

        return predicted_stock_price, actual_stock_price

    def evaluate_model(self, actual, predicted):
        """
        Evaluate the model using MAE, RMSE, and MAPE metrics.
        """
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return mae, rmse, mape


    def optimize_arima(self):
        """
        Optimize ARIMA model parameters using auto_arima from pmdarima.
        """
        model = auto_arima(self.data[f'Close {self.ticker}'], seasonal=True, m=5, trace=True, suppress_warnings=True)
        self.best_arima_model = model.fit(self.data[f'Close {self.ticker}'])


    def compare_models(self):
        """
        Compare the forecasts from all models.
        """
        print("Evaluating LSTM Model:")
        predicted_lstm, actual_lstm = self.forecast_lstm()
        lstm_mae, lstm_rmse, lstm_mape = self.evaluate_model(actual_lstm, predicted_lstm)
        print(f"LSTM Model - MAE: {lstm_mae}, RMSE: {lstm_rmse}, MAPE: {lstm_mape}")

        print("Evaluating ARIMA Model:")
        arima_forecast = self.forecast_arima()
        arima_mae, arima_rmse, arima_mape = self.evaluate_model(self.test_data[f'Close {self.ticker}'], arima_forecast)
        print(f"ARIMA Model - MAE: {arima_mae}, RMSE: {arima_rmse}, MAPE: {arima_mape}")




