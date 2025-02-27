"""Data downloading, preprocessing and portfolio analysis."""

import yfinance as yf




class PortfolioAnalysis():
  def __init__(self, tickers, start_date, end_date):

    self.tickers = tickers
    self.start_date = start_date
    self.end_date = end_date
    self.date = self.fetch_data()


  def fetch_data(self):
    """
    Fetch the historical stock data for each ticker from Yahoo Finance.
    """
    data = yf.download(self.tickers, self.start_date, self.end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    return data
