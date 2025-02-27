"""Data downloading, preprocessing and portfolio analysis."""

import yfinance as yf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"


class PortfolioAnalysis():
    def __init__(self, tickers, start_date, end_date):

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()

    def _fetch_data(self):
        """
        Fetch the historical stock data for each ticker from Yahoo Finance.
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)

        # Flatten the multi-index columns
        data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

        # Select only the required columns
        selected_columns = [col for col in data.columns if any(field in col for field in ['Open', 'High', 'Low', 'Close', 'Volume'])]
        data = data[selected_columns]

        return data

    def save_raw_data(self, save_path):
        """Saving raw data."""
        self.data.to_csv(save_path)
        print(f"Raw data saved successfully as {save_path}.")

    def clean_data(self):
        """
        Clean the data by checking for missing values, ensuring correct data types,
        and handling missing values.
        """
        print(f"Data downloaded columns\n{self.data.columns}\n")
        print(f"Data downloaded shape\n{self.data.shape}")

        # Check for missing values
        missing_data = self.data.isnull().sum()
        print(f"Missing Data:\n{missing_data}\n")

        # Handle missing values (use .ffill() instead of deprecated method)
        self.data.ffill(inplace=True)

        # Get all tickers and generate proper column names
        tickers = self.tickers if isinstance(
            self.tickers, list) else [self.tickers]
        columns_to_convert = []

        for ticker in tickers:
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                col_name = f"{col} {ticker}"
                if col_name in self.data.columns:
                    columns_to_convert.append(col_name)

        # Convert selected columns to the appropriate type
        self.data[columns_to_convert] = self.data[columns_to_convert].astype(
            float)

    def normalize_data(self):
        """
        Normalize the data using MinMaxScaler for machine learning models.
        """
        scaler = MinMaxScaler()

        # Identify ticker-specific column names dynamically
        tickers = self.tickers if isinstance(self.tickers, list) else [self.tickers]
        columns_to_normalize = [col for col in self.data.columns if any(field in col for field in ["Open", "High", "Low", "Close"])]

        # Apply MinMaxScaler only on available columns
        if columns_to_normalize:
            self.data[columns_to_normalize] = scaler.fit_transform(self.data[columns_to_normalize])
        else:
            print("⚠️ No matching columns found for normalization.")

    def save_preprocessed_data(self, save_path):
        """Saving preprocessed data."""
        self.data.to_csv(save_path)
        print(f"Preprocessed data saved successfully as {save_path}.")

    def perform_eda(self):
        """
        Perform Exploratory Data Analysis including visualization and volatility analysis.
        """
        self.visualize_data()
        self.calculate_daily_pct_change()
        self.analyze_volatility()
        self.detect_outliers()

        print("Performing EDA completed successfully!")
        print(f"\n{'*'*100}\n")

    def visualize_data(self):
        """
        Visualize the closing price and other key metrics.
        """
        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            col_name = f"Close {ticker}"
            if col_name in self.data.columns:
                self.data[col_name].plot(label=f'{ticker} Close Price')
            else:
                print(f"⚠️ {col_name} not found in data!")

        plt.title('Closing Price of Assets Over Time')
        plt.legend()
        plt.savefig(f'{BASE_DIR}/plots/eda/closing_price.png')
        plt.grid(True)
        plt.show()


    def calculate_daily_pct_change(self):
        """
        Calculate daily percentage change to observe volatility.
        """
        close_columns = [col for col in self.data.columns if "Close" in col]

        if close_columns:
            daily_pct_change = self.data[close_columns].pct_change() * 100
            plt.figure(figsize=(12, 6))
            daily_pct_change.plot()
            plt.title('Daily Percentage Change in Closing Price')
            plt.legend()
            plt.savefig(f'{BASE_DIR}/plots/eda/daily_pct_change.png')
            plt.grid(True)
            plt.show()
            return daily_pct_change
        else:
            print("⚠️ No 'Close' price columns found!")
            return None


    def analyze_volatility(self):
        """
        Analyze the volatility by calculating rolling means and standard deviations.
        """
        rolling_window = 30  # 30 days rolling window for volatility analysis
        for ticker in self.tickers:
            col_name = f"Close {ticker}"
            if col_name in self.data.columns:
                rolling_mean = self.data[col_name].rolling(window=rolling_window).mean()
                rolling_std = self.data[col_name].rolling(window=rolling_window).std()

                plt.figure(figsize=(12, 6))
                self.data[col_name].plot(label=f'{ticker} Close Price')
                rolling_mean.plot(label=f'{ticker} 30-day Rolling Mean', linestyle='--')
                rolling_std.plot(label=f'{ticker} 30-day Rolling Std Dev', linestyle=':')
                plt.title(f'{ticker} Volatility and Rolling Statistics')
                plt.legend()
                plt.savefig(f'{BASE_DIR}/plots/eda/volatility_{ticker}.png')
                plt.grid(True)
                plt.show()
            else:
                print(f"⚠️ {col_name} not found in data!")


    def detect_outliers(self):
      """
      Detect outliers in the daily returns using z-scores.
      """
      daily_pct_change = self.calculate_daily_pct_change()
      z_scores = zscore(daily_pct_change.dropna())
      outliers = np.where(np.abs(z_scores) > 3)
      print(f"Outliers detected at indices: {outliers}")

      # Plot the boxplot
      plt.figure(figsize=(12, 6))
      sns.boxplot(data=daily_pct_change, orient='h', width=0.5)
      plt.title("Boxplot of Daily Percentage Change (Detecting Outliers)")
      plt.xlabel("Daily Percentage Change")
      plt.savefig(f'{BASE_DIR}/plots/eda/boxplot_outliers.png')
      plt.show()


    def decompose_time_series(self):
        """
        Decompose the time series into trend, seasonal, and residual components.
        """
        print(f"\n{'*'*100}\n")
        for ticker in self.tickers:
            series = self.data[f'Close {ticker}']

            print("\n")

            # Check for zero and negative values
            if np.any(series <= 0):
                print(f"⚠️ {ticker} contains zero or negative values! Using additive model instead.\n")
                decomposition = seasonal_decompose(series, model='additive', period=252)
            else:
                decomposition = seasonal_decompose(series, model='multiplicative', period=252)

            decomposition.plot()
            plt.title(f'{ticker} Time Series Decomposition')
            plt.legend()
            plt.savefig(f'{BASE_DIR}/plots/eda/time_series_decomposition_{ticker}.png')
            plt.show()

            print(f"Time series decomposition for {ticker} completed successfully!\n")

    def calculate_key_metrics(self):
        """
        Calculate key financial metrics like VaR (Value at Risk) and Sharpe Ratio.
        """
        daily_returns = self.calculate_daily_pct_change()
        # Value at Risk (VaR) - 1-day 99% confidence level
        var_99 = np.percentile(daily_returns.dropna(), 1)
        print(f"1-Day VaR at 99% Confidence Level: {var_99}%")

        # Sharpe Ratio - Assuming risk-free rate is 0
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        print(f"Sharpe Ratio:\n {sharpe_ratio}")

    def summarize_insights(self):
        """
        Document key insights like overall trends and risk metrics.
        """
        self.calculate_key_metrics()
        print(f"\n{'*'*100}\n")
        print("Key Insights:")
        print(f"Overall direction of Tesla's stock price: {'Up' if self.data['Close TSLA'].iloc[-1] > self.data['Close TSLA'].iloc[0] else 'Down'}")
        print("Fluctuations in daily returns and their impact have been considered in VaR and Sharpe ratio calculations.")

        print(f"\n{'*'*100}\n")
        print("Insights summarized successfully!")

