"""Process pipeliner that rins processes end-to-end."""

import os
import sys

import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

from scripts.data_analysis import PortfolioAnalysis
from scripts.stock_forecasting import StockForecasting


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week-11/portfolio-management-optimization"

if __name__ == "__main__":

    # Pipeline to run data_analysis processes

    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-01-01'
    end_date = '2025-01-31'

    # Instantiate and run the Portfolio Analysis
    portfolio_analysis = PortfolioAnalysis(tickers, start_date, end_date)
    print(f"\n{'*'*100}\n")
    portfolio_analysis.save_raw_data(f"{BASE_DIR}/data/raw_data.csv")
    print(f"\n{'*'*100}\n")
    portfolio_analysis.clean_data()
    print("Handled data missing and Ensured data types of each data")
    print(f"\n{'*'*100}\n")
    portfolio_analysis.normalize_data()
    print("Normalized Data")
    print(f"\n{'*'*100}\n")
    save_path = f"{BASE_DIR}/data/preprocessed_data.csv"
    portfolio_analysis.save_preprocessed_data(save_path)
    print(f"\n{'*'*100}\n")
    portfolio_analysis.perform_eda()
    print(f"\n{'*'*100}\n")
    portfolio_analysis.decompose_time_series()
    print(f"\n{'*'*100}\n")

    portfolio_analysis.summarize_insights()

    # Pipeline to run stock forcasting processes
    preprocessed_data = pd.read_csv(f"{BASE_DIR}/data/preprocessed_data.csv", index_col=0)
    ticker = "TSLA"
    stock_forecasting = StockForecasting(preprocessed_data, ticker)
    stock_forecasting.retrieve_data_by_ticker(ticker, inplace=True)
    stock_forecasting.split_data()

    # Train models
    stock_forecasting.arima_model()
    stock_forecasting.sarima_model()
    stock_forecasting.train_lstm(look_back=60, epochs=10, batch_size=32)

    # Optimize ARIMA
    stock_forecasting.optimize_arima()


    # Save all models
    stock_forecasting.save_all_models()

    # Compare models
    stock_forecasting.compare_models()
