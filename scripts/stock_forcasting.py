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
    def __init__(self, processed_data):
        self.data = processed_data


    def retrieve_data_by_ticker(self, ticker, inplace=False):
        """Will filter data by tickers."""
        if ticker in self.data.columns:
            # Filter columns that contain the ticker name (e.g., 'Tsla')
            filter_columns = [col for col in self.data.columns if ticker in col]

            # If inplace is True, update self.data in place
            if inplace:
                self.data = self.data[filter_columns]
            else:
                # If inplace is False, return the filtered data
                return self.data[filter_columns]
        else:
            raise ValueError(f"Ticker {ticker} not found in data columns.")




    def split_data(self):
        """Split into train and test sets (80% train, 20% test)."""
        train_size = int(len(self.data) * 0.8)
        self.train_data, self.test_data = self.data[:train_size], self.data[train_size:]

        self.train_data = pd.DataFrame(self.train_data, columns=['Close'])
        self.test_data = pd.DataFrame(self.test_data, columns=['Close'])


    def arima_model(self):
        """
        Fit an ARIMA model on the stock data.
        """
        model = ARIMA(self.data['Close'], order=(5, 1, 0))  # Example parameters
        self.arima_model = model.fit()

    def forecast_arima(self):
        """
        Forecast future stock prices using the ARIMA model.
        """
        forecast = self.arima_model.forecast(steps=len(self.test_data))
        return forecast

    def sarima_model(self):
        """
        Fit a SARIMA model on the stock data.
        """
        model = SARIMAX(self.data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 5))  # Example parameters
        self.sarima_model = model.fit()

    def forecast_sarima(self):
        """
        Forecast future stock prices using the SARIMA model.
        """
        forecast = self.sarima_model.forecast(steps=len(self.test_data))
        return forecast


    def create_lstm_data(self, look_back=60):
        """
        Prepare the data for LSTM by creating time series sequences.
        """
        X_train, y_train, X_test, y_test = [], [], [], []

        for i in range(look_back, len(self.train_data)):
            X_train.append(self.train_data['Close'].iloc[i-look_back:i].values)
            y_train.append(self.train_data['Close'].iloc[i])

        for i in range(look_back, len(self.test_data)):
            X_test.append(self.test_data['Close'].iloc[i-look_back:i].values)
            y_test.append(self.test_data['Close'].iloc[i])

        # Reshape for LSTM input
        X_train, X_test = np.array(X_train), np.array(X_test)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test


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

    def train_lstm(self, look_back=60, epochs=5, batch_size=32):
        """
        Train the LSTM model.
        """
        X_train, y_train, X_test, y_test = self.create_lstm_data(look_back)
        model = self.build_lstm_model(look_back)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self.lstm_model = model
        self.X_test, self.y_test = X_test, y_test

    def forecast_lstm(self):
        """
        Forecast future stock prices using the trained LSTM model.
        """
        predicted_stock_price = self.lstm_model.predict(self.X_test)
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)
        actual_stock_price = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

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
        model = auto_arima(self.data['Close'], seasonal=True, m=5, trace=True, suppress_warnings=True)
        self.best_arima_model = model.fit(self.data['Close'])




