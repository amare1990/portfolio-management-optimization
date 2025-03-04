"""Portfolio optimization."""
import pandas as pd

from scripts.forecasting_future_market_trends import ForecastFutureMarkets

BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"

class PortfolioOptimization:
    def __init__(self, model, tickers):
        self.model = model
        self.tickers = tickers
        self.forecasts = {}  # Dictionary to store forecasts for each ticker

    def generate_forecasts(self):
        for ticker in self.tickers:
            forecaster = ForecastFutureMarkets(ticker, f"{BASE_DIR}/data/preprocessed_data.csv")
            self.forecasts[ticker] = forecaster.forecast_arima()  # Assuming this returns a DataFrame

    def merge_forecasts(self):
        if not self.forecasts:
            raise ValueError("No forecasts available. Run generate_forecasts() first.")

        # Merge all forecast DataFrames on the Date index
        merged_df = pd.concat(self.forecasts.values(), axis=1, keys=self.forecasts.keys())
        return merged_df


