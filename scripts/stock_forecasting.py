"""Time-series forcasting using statistical and ML models."""

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import pickle


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"

# Define the StockForecasting class
class StockForecasting:
    def __init__(self, processed_data, ticker):
        self.data = processed_data
        self.scaled = pd.DataFrame(index=self.data.index)
        self.ticker = ticker
        self.scaler = MinMaxScaler()



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



    def normalize_data(self):
        """
        Normalize the data using MinMaxScaler for machine learning models.
        """
        columns_to_normalize = [f'Close {self.ticker}']
        # Apply MinMaxScaler only on available columns
        if columns_to_normalize:
            self.scaled[columns_to_normalize] = self.scaler.fit_transform(self.data[columns_to_normalize])
        else:
            print("⚠️ No matching columns found for normalization.")


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
        return pd.Series(forecast, index=self.test_data.index, name=f'Close {self.ticker}')


    def sarima_model(self):
        """
        Fit a SARIMA model on the stock data.
        """
        model = SARIMAX(self.data[f'Close {self.ticker}'], order=(5, 1, 0), seasonal_order=(1, 1, 0, 12))
        self.sarima_model = model.fit()

    def forecast_sarima(self):
        """
        Forecast future stock prices using the SARIMA model.
        """
        forecast = self.sarima_model.forecast(steps=len(self.test_data))
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
            X.append(self.scaled.iloc[i:(i + look_back), 0].values)
            y.append(self.scaled.iloc[i + look_back, 0])
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

        # Change the scaled predicted data into the unscaled predicted data
        predicted_stock_price = self.scaler.inverse_transform(predicted_stock_price)

        actual_stock_price = self.y_test.reshape(-1, 1)

         # Change the scaled actual data into the original actual data
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
        model = auto_arima(self.data[f'Close {self.ticker}'], seasonal=True, m=5, trace=True, suppress_warnings=True)
        self.best_arima_model = model.fit(self.data[f'Close {self.ticker}'])

    def save_model(self, model, filename):
        """
        Save a model using pickle (for ARIMA and SARIMA) or TensorFlow (for LSTM).
        """
        if isinstance(model, tf.keras.Model):
            model.save(f"{BASE_DIR}/models/{filename}")  # Save LSTM model using TensorFlow
        else:
            with open(f"{BASE_DIR}/models/{filename}", 'wb') as file:
                pickle.dump(model, file)
        print(f"Model saved successfully as {BASE_DIR}/models/{filename}")



    def save_all_models(self):
        """
        Save all trained models.
        """
        self.save_model(self.arima_model, "arima_model.pkl")
        self.save_model(self.sarima_model, "sarima_model.pkl")
        self.save_model(self.best_arima_model, "optimized_arima_model.pkl")
        self.save_model(self.lstm_model, "lstm_model.h5")



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




