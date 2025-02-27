"""Data downloading, preprocessing and portfolio analysis."""

import yfinance as yf

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


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
        data = yf.download(self.tickers, self.start_date, self.end_date)
        # Select columns and flatten the multi-index columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Flatten columns
        data.columns = data.columns.map(' '.join).str.strip()
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
        tickers = self.tickers if isinstance(
            self.tickers, list) else [self.tickers]
        columns_to_normalize = []

        for ticker in tickers:
            for col in ["Open", "High", "Low", "Close"]:
                col_name = f"{col} {ticker}"
                if col_name in self.data.columns:
                    columns_to_normalize.append(col_name)

        # Apply MinMaxScaler only on available columns
        if columns_to_normalize:
            self.data[columns_to_normalize] = scaler.fit_transform(
                self.data[columns_to_normalize])
        else:
            print("⚠️ No matching columns found for normalization.")

    def save_preprocessed_data(self, save_path):
        """Saving preprocessed data."""
        self.data.to_csv(save_path)
        print(f"Preprocessed data saved successfully as {save_path}.")
