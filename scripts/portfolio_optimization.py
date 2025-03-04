"""Portfolio optimization for all the three assests where ARIMA statistical mdoel
is used to generate forecast data within 6-12 months."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
        daily_returns = self.returns.mean() * 252  # Assuming 252 trading/working days in a year
        return daily_returns

    def calculate_covariance_matrix(self):
        """
        Compute the covariance matrix of returns.
        """
        return self.returns.cov() * 252  # Annualize the covariance matrix


    def portfolio_performance(self, weights):
        """
        Calculate the expected portfolio return and volatility.
        """
        # Expected portfolio return
        portfolio_expected_return = np.sum(self.calculate_annual_return() * weights)

        # Expected portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.calculate_covariance_matrix(), weights)))

        return portfolio_expected_return, portfolio_volatility


    def negative_sharpe_ratio(self, weights, risk_free_rate=0.0):
        """
        Objective function to minimize the negative Sharpe Ratio.
        """
        portfolio_expected_return, portfolio_volatility = self.portfolio_performance(weights)
        return -(portfolio_expected_return - risk_free_rate) / portfolio_volatility


    def optimize_portfolio(self):
        """
        Optimize portfolio by maximizing the Sharpe Ratio.
        """
        # Initial guess for portfolio weights (equal allocation)
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets

        # Constraints: weights should sum to 1, and each weight should be between 0 and 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))

        # Optimize the portfolio
        result = minimize(self.negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x


    def calculate_var(self, portfolio_weights, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) for the portfolio.
        """
        portfolio_return, portfolio_volatility = self.portfolio_performance(portfolio_weights)

        # Calculate the z-score for the given confidence level
        z_score = np.percentile(np.random.normal(0, 1, 100000), (1 - confidence_level) * 100)

        # Calculate VaR at the given confidence level
        var = portfolio_return - z_score * portfolio_volatility
        return var


    def visualize_portfolio(self, optimal_weights):
        """
        Visualize the portfolio performance.
        """
        portfolio_return, portfolio_volatility = self.portfolio_performance(optimal_weights)
        print(f"Optimized Portfolio  Expected Return: {portfolio_return * 100:.2f}%")
        print(f"Optimized Portfolio Volatility: {portfolio_volatility * 100:.2f}%")

        # Simulate portfolio growth over time
        portfolio_growth = (self.returns.dot(optimal_weights) + 1).cumprod()

        # Plot cumulative return
        plt.figure(figsize=(14,7))
        plt.plot(portfolio_growth, label="Optimized Portfolio Growth", color='green')
        plt.title("Portfolio Performance (Cumulative Return)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.savefig(
            f'{BASE_DIR}/notebooks/plots/optimization/portfolio_performance.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
