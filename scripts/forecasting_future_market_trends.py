"""USing the already trained price forecasting model."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pickle
from tensorflow.keras.models import load_model


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"


class Forecast_Future_Markets:
  def __init__(self, processed_file_path):
    self.data = pd.read_csv(processed_file_path)


  def load_model(self, filename):
        """
        Load a model using pickle (for ARIMA and SARIMA) or TensorFlow (for LSTM).
        """
        try:
            if filename.endswith('.h5'):
                model = load_model(filename)  # Load LSTM model using TensorFlow
            else:
                with open(filename, 'rb') as file:
                    model = pickle.load(file)
            print(f"Model loaded successfully from {BASE_DIR}/models/{filename}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


  def load_all_models(self):
        """
        Load all saved models.
        """
        self.arima_model = self.load_model(f"{BASE_DIR}/models/arima_model.pkl")
        self.sarima_model = self.load_model(f"{BASE_DIR}/models/sarima_model.pkl")
        self.best_arima_model = self.load_model(f"{BASE_DIR}/models/optimized_arima_model.pkl")
        self.lstm_model = self.load_model(f"{BASE_DIR}/models/lstm_model.h5")


  def forecast_arima(self, steps=180):
        """ Forecast future stock prices using the ARIMA model. """
        forecast = self.arima_model.forecast(steps=steps)

        print("Forecasting using ARIMA completed successfully!")
        print(f"\n{'='*100}")

        return pd.Series(forecast, name="ARIMA Forecast")

  def forecast_sarima(self, steps=180):
        """ Forecast future stock prices using the SARIMA model. """
        forecast = self.sarima_model.get_forecast(steps=steps)
        conf_int = forecast.conf_int()

        print("Forecasting using SARIMA completed successfully!")
        print(f"\n{'='*100}")

        return forecast.predicted_mean, conf_int


  def forecast_lstm(self, steps=180):
        """ Forecast future stock prices using the trained LSTM model. """
        if self.lstm_model is None:
            print("LSTM model is not loaded. Please load the model first.")
            return None

        inputs = self.data[f'Close {self.ticker}'].values.reshape(-1, 1)  # Ensure input is reshaped correctly

        lstm_forecast = []

        for _ in range(steps):
            X_input = inputs[-60:].reshape((1, 60, 1))  # Ensure correct shape for LSTM
            pred_price = self.lstm_model.predict(X_input)[0, 0]
            lstm_forecast.append(pred_price)
            inputs = np.append(inputs, pred_price).reshape(-1, 1)  # Append new prediction

        # ✅ No need for inverse_transform since data is already standardized
        lstm_forecast = np.array(lstm_forecast).reshape(-1, 1)

        print("Forecasting using LSTM completed successfully!")
        print(f"\n{'='*100}")

        return lstm_forecast

  def visualize_forecast(self, forecast, model_name="Model"):
        """ Visualize forecast alongside historical data. """
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data[f'Close {self.ticker}'], label="Historical Data", color='blue')
        # Generate date range for forecast
        forecast_index = pd.date_range(start=self.data.index[-1], periods=len(forecast) + 1, freq='D')[1:]
        plt.plot(forecast_index, forecast, label=f"Forecast - {model_name}", color='red')
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.savefig(
            f"{BASE_DIR}/plots/market_trends/{model_name}_forecast.png",
            dpi=300,
            bbox_inches='tight'
            )
        plt.show()
