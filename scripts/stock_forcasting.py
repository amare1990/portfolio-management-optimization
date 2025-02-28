"""Time-series forcasting using statistical and ML models."""

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

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




