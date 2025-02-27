"""Data downloading, preprocessing and portfolio analysis."""

import yfinance as yf




class PortfolioAnalysis():
  def __init__(self, tickers, start_date, end_date):

    self.tickers = tickers
    self.start_date = start_date
    self.end_date = end_date
    self.date = self._fetch_data()


  def _fetch_data(self):
    """
    Fetch the historical stock data for each ticker from Yahoo Finance.
    """
    data = yf.download(self.tickers, self.start_date, self.end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

  def clean_data(self):
        """
        Clean the data by checking for missing values, ensuring correct data types,
        and handling missing values.
        """
        # Check for missing values and handle them
        missing_data = self.data.isnull().sum()
        print(f"Missing Data:\n{missing_data}\n")

        # Handle missing values - filling with forward fill for simplicity
        self.data.fillna(method='ffill', inplace=True)

        # Ensure all columns have appropriate data types
        self.data = self.data.astype({
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            # 'Adj Close': 'float64',
            'Volume': 'int64'
        })
