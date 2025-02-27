"""Data downloading, preprocessing and portfolio analysis."""






class PortfolioAnalysis():
  def __init__(self, tickers, start_date, end_date):

    self.tickers = tickers
    self.start_date = start_date
    self.end_date = end_date
    self.date = self.fetch_data()

