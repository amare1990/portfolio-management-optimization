"""USing the already trained price forecasting model."""

import numpy as np
import pandas as pd

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
