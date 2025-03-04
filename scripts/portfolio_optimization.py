"""Portfolio optimization for all the three assests where ARIMA statistical mdoel
is used to generate forecast data within 6-12 months."""

import pandas as pd
from scripts.forecasting_future_market_trends import ForecastFutureMarkets
from scripts.stock_forecasting import StockForecasting

BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"

class PortfolioOptimization:
    def __init__(self, tickers, start_date="2024-02-01"):
        self.tickers = tickers
        self.forecasts = {}  # Dictionary to store forecasts for each ticker
        self.start_date = start_date  # Set the forecast start date
        self.data = None

    def generate_forecasts(self):
        for ticker in self.tickers:
            print(f"Generating forecast for: {ticker}")

            preprocessed_data = pd.read_csv(f"{BASE_DIR}/data/preprocessed_data.csv", index_col=0)
            preprocessed_data.index = pd.to_datetime(preprocessed_data.index)

            stock_forecasting = StockForecasting(preprocessed_data, ticker)
            stock_forecasting.retrieve_data_by_ticker(ticker, inplace=True)

            # Train ARIMA model and store it
            stock_forecasting.arima_model()
            arima_model = stock_forecasting.arima_model
            if arima_model is None:
                print(f"Error: ARIMA model for {ticker} is None.")
                continue

            stock_forecasting.save_model(arima_model, "arima_model.pkl")
            forecaster = ForecastFutureMarkets(ticker, f"{BASE_DIR}/data/preprocessed_data.csv")
            forecaster.load_all_models()

            forecast = forecaster.forecast_arima()

            print(f"Forecast for {ticker}:\n", forecast.head())

            self.forecasts[ticker] = forecast



    def merge_forecasts(self):
        if not self.forecasts:
            raise ValueError("No forecasts available. Run generate_forecasts() first.")

        # Merge forecasts into a single DataFrame
        merged_df = pd.concat(self.forecasts.values(), axis=1, keys=self.forecasts.keys())

        # Convert numerical index to datetime
        forecast_horizon = merged_df.shape[0]  # Number of forecasted days
        merged_df.index = pd.date_range(start=self.start_date, periods=forecast_horizon, freq="B")  # 'B' for business days
        merged_df.index.name = 'Date'
        self.data = merged_df

        return merged_df


    def _calculate_returns(self):
        """
        Calculate daily returns for the assets.
        """
        returns = self.data.pct_change().dropna()
        return returns

    def calculate_annual_return(self):
        """
        Compute the annual return for each asset.
        """
        daily_returns = self.returns.mean() * 252  # Assuming 252 trading days in a year
        return daily_returns
