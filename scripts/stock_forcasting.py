"""Time-series forcasting using statistical and ML models."""

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define the StockForecasting class
class StockForecasting:
    def __init__(self, processed_data):
        self.data = processed_data


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




