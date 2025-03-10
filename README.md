# Portfolio Management Optimization

> Portfolio Management Optimization is a data science project aimed at analyzing and optimizing a financial portfolio using historical stock market data. It enables users to download and preprocess stock data from Yahoo Finance, perform exploratory data analysis (EDA) to gain insights, conduct volatility analysis and outlier detection, compute financial metrics like Value at Risk (VaR) and Sharpe Ratio, and summarize key insights and trends. it also builds statistical andmodels like ARIMA and Deep Learning models like LSTSM using historical price data. It then forecasts future market trends and optimze portfolio based on the forecast future market prices. This project is implemented in Python and leverages powerful financial libraries.

## Built With

- Programming Language: Python 3
- Libraries: NumPy, pandas, matplotlib, Seaborn, scipy, statsmodels, etc
- Tools & Technologies: Jupyter Notebook, Google Colab, Git, GitHub, Gitflow, VS Code

## Demonstration and Website

[Deployment link]()


---
## Project Structure
```bash
portfolio-management-optimization/
│── .github/workflows/              # GitHub Actions for CI/CD
│   ├── pylint.yml                 # Workflow for checking Python code styles
│── data/                          # Directory for datasets
│   ├── raw_data.csv         # Raw dataset downloaded using yfinance
│   ├── preprocessed_data.csv         # Processed dataset
│
│── notebooks/                     # Jupyter notebooks for analysis
│   ├── data_analyzer.ipynb         # Notebook for data analysis
    ├── stock_forecaster.ipynb         # Notebook for building models and stock price forecasting
    ├── market_trends_forecaster.ipynb         # Notebook for future market trends forecasting
│   ├── plots/eda                      # Directory for storing generated plots while performing EDA
│       ├── closing_prie.png    # Time series plot
│       ├── daily_pct_change.png
│
│── scripts/                        # Python scripts for different modules
    ├── __init__.py
│   ├── data_analysis.py            # Data downloading, cleaning, analysis, normalization, and EDA
│   ├── stock_forecatsing.py            # Forecasting closing prices of TSLA for the next 30 days using ARIMA, SARIMA, Optimized_ARIMA and LSTM
│   ├── forecasting_future_market_trends.py            # Forecasting closing prices of TSLA for the next 6-12 months using ARIMA, SARIMA, LSTM
│   ├── portfolio_optimization.py            # Portfolio optimization

│
│── main/                            # Main automation script
    ├── __init__.py
│   ├── src.py                      # Main pipeline script
│
│── tests/              # Folder that contain testing scripts
│   ├── __init__.py
│
│── requirements.txt                 # Dependencies and libraries
│── README.md                        # Project documentation
│── .gitignore                        # Files and directories to ignore in Git
```

---

## Getting Started

You can clone this project and use it freely. Contributions are welcome!

### Cloning the Repository

To get a local copy, run the following command in your terminal:

```sh
git clone https://github.com/amare1990/portfolio-management-optimization.git
```

Navigate to the main directory:

```sh
cd portfolio-management-optimization
```

### Setting Up a Virtual Environment

1. Create a virtual environment:
   ```sh
   python -m venv venv-name
   ```
   Replace `venv-name` with your desired environment name.

2. Activate the virtual environment:
   - **On Linux/macOS:**
     ```sh
     source venv-name/bin/activate
     ```
   - **On Windows:**
     ```sh
     venv-name\Scripts\activate
     ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Project

- **To execute the full pipeline automatically:**
  ```sh
  python main/main.py
  ```
  This runs the entire workflow end-to-end without manual intervention.

- **To run and experiment with individual components:**
  1. Open Jupyter Notebook:
     ```sh
     jupyter notebook
     ```
  2. Run each notebook. The notebooks are named to match their corresponding scripts for easy navigation.
  3. You can also run each function manually to inspect intermediate results.

---

## Prerequisites

Ensure you have the following installed:

- Python (minimum version **3.8.10**)
- pip
- Git
- VS Code (or another preferred IDE)

---

## Dataset

The dataset used in this project is obtained from **Yahoo Finance** and processed as follows:

- **Tickers:** `['TSLA', 'BND', 'SPY']`
- **Date Range:** `'2015-01-01'` to `'2025-01-31'`
- **Data Retrieval:**
```python
import yfinance as yf

tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-01-01'
end_date = '2025-01-31'
data = yf.download(tickers, start=start_date, end=end_date)

# Flatten multi-index columns
data.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
```

## Project Requirements

### 1. GitHub Actions and Python Coding Standards

- Automated code checks: The repository includes Pylint linters in `.github/workflows` to maintain coding standards.
- Pull Request validation: The linter automatically runs and checks for errors whenever a pull request is created

#### Manual Linting Commands

- Check code formatting in the `scripts/` directory:
  ```sh
  pylint scripts/*.py
  ```

- Auto-fix linter errors in the `scripts/` directory:
  ```sh
  autopep8 --in-place --aggressive --aggressive scripts/*.py
  ```

- Check code formatting in the `main/` directory (main processing pipeline):
  ```sh
  pylint main/main.py
  ```

- Auto-fix linter errors in the `main/` directory:
  ```sh
  autopep8 --in-place --aggressive --aggressive main/main.py
  ```

---



### 2. Preprocess and Explore the Data
This section is handled by the `PortfolioAnalysis` class in `data_analysis.py`. It performs essential loading, preprocessing, and exploratory data analysis (EDA) on the data downloaded using `yfinance` from companies with tickers `TSLA, BND, SPY`. The key steps include:

#### 2.1 Data Downloading
- Fetches historical stock data from Yahoo Finance.
- Selects key columns such as `Open`, `High`, `Low`, `Close`, and `Volume`.
- Saves the raw data for further processing.

#### 2.2 Data Cleaning
- Checks for missing values and fills them using forward fill (`.ffill()`).
- Converts necessary columns to appropriate data types.

#### 2.3 Data Normalization
- Uses `MinMaxScaler` to normalize selected stock features (`Open`, `High`, `Low`, `Close`).

#### 2.4 Exploratory Data Analysis (EDA)
- Visualizes stock closing prices over time.
- Computes daily percentage changes to analyze volatility.
- Analyzes volatility using rolling means and standard deviations.
- Detects outliers in daily returns using z-scores.
- Performs time series decomposition to extract trend, seasonality, and residual components.

#### 2.5. Financial Metrics Calculation
This module computes key financial risk and return metrics to evaluate portfolio performance:

#### 2.5.1 Value at Risk (VaR)
- Calculates the 1-day 99% confidence level VaR to assess potential losses.

#### 2.5.2 Sharpe Ratio
- Measures the risk-adjusted return assuming a risk-free rate of zero.

**Note:** If the Sharpe Ratio returns NaN, ensure there are no missing or constant daily returns in the dataset.

#### 2.6. Insights and Summary
After processing the data and calculating key financial metrics, the system:
- Summarizes overall stock trends.
- Analyzes portfolio risk and return using VaR and Sharpe Ratio.
- Provides actionable insights on stock market performance.

#### 2.7. How to Use
##### 2.7.1 Requirements
Ensure you have the following dependencies installed:
```bash
pip install yfinance numpy pandas matplotlib seaborn scikit-learn statsmodels
```

##### 2.7.2 Running the Analysis
- Open Jupyter notebook, open `data_analyzer.ipynb` and run it:
- You can see the outputs from end-to-end
```python
python main/main.py

```

#### 2.8 Expected Outputs
- Processed CSV files with cleaned and normalized stock data.
- Visualizations of price trends, volatility, and outliers.
- Computed VaR and Sharpe Ratio for risk assessment.



### 2. Stock Forecasting

This section covers the implementation of time-series forecasting using various statistical and machine learning models.
The primary goal is to predict stock prices based on historical data using methods like ARIMA, SARIMA, and LSTM.

#### Overview of StockForecasting Class

The `StockForecasting` class enables forecasting of stock prices using different approaches.
It integrates data preprocessing, model training, and evaluation functionalities.
The class supports the following models:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: A traditional statistical method for time-series forecasting.
2. **SARIMA (Seasonal ARIMA)**: An extension of ARIMA that accounts for seasonal patterns in the data.
3. **LSTM (Long Short-Term Memory)**: A deep learning model that is particularly effective in capturing patterns in sequential data.

#### Key Methods

- **retrieve_data_by_ticker(ticker, inplace=False)**:
  This method filters the stock data by ticker and renames columns appropriately for further processing.

- **split_data()**:
  Splits the data into training and testing sets (80% training, 20% testing), focusing on the 'Close' price for the given ticker.

- **arima_model() & forecast_arima()**:
  Fits an ARIMA model to the stock data and forecasts future stock prices.

- **sarima_model() & forecast_sarima()**:
  Fits a SARIMA model and forecasts future prices, accounting for seasonal variations.

- **build_lstm_model()**:
  Constructs an LSTM model for stock price prediction with adjustable input size and layers.

- **create_lstm_data()**:
  Prepares the data to be fed into the LSTM model by creating sequences of past stock prices.

- **train_lstm()**:
  Trains the LSTM model using the prepared data and specified training parameters (e.g., look-back period, number of epochs, and batch size).

- **forecast_lstm()**:
  Makes stock price predictions using the trained LSTM model.

- **evaluate_model()**:
  Evaluates the model performance using several metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

- **optimize_arima()**:
  Optimizes ARIMA parameters using the `auto_arima` function from the `pmdarima` package to improve forecast accuracy.

- **compare_models()**:
  Compares the forecast performance of all models (ARIMA, SARIMA, and LSTM) by calculating and displaying MAE, RMSE, and MAPE for each.

#### Usage

- To run and view each process of the stcok foracsting, open stock_forcaster.ipnb and run it
```python
To see the stock_forecast result from the end -to-end result
python main/main.py

```

#### Evaluation

Model evaluation is performed using three key metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared differences between predicted and actual values.
- **Mean Absolute Percentage Error (MAPE)**: Measures the average percentage difference between predicted and actual values.

---


### 3. Forecast Future Market Trends

This module leverages various machine learning models to forecast future market trends based on historical stock data. It loads pre-trained models (ARIMA, SARIMA and LSTM) to predict future stock prices. The models are designed to forecast price movements for the next 180 days. The corresponding notebook to run this feature is `market_trends_forecaster.ipynb`.

#### Forecast Models:
1. **ARIMA (AutoRegressive Integrated Moving Average)**: A time series forecasting method that uses past values to predict future values. This model is suitable for univariate data where patterns can be detected and extrapolated.
2. **SARIMA (Seasonal ARIMA)**: An extension of ARIMA that accounts for seasonality in the data, making it ideal for financial time series data with seasonal trends.
3. **LSTM (Long Short-Term Memory)**: A deep learning model specifically designed to handle sequences of data and learn long-term dependencies. LSTM models are effective for forecasting stock prices due to their ability to learn from temporal data.

#### Key Features:
- **Model Loading**: All models are loaded via a dedicated function `load_model()`, which supports both ARIMA/SARIMA (using pickle) and LSTM (using TensorFlow).
- **Forecasting**: The `forecast_arima()`, `forecast_sarima()`, and `forecast_lstm()` methods perform predictions based on the selected model. The forecasts are generated for the next 180 steps (days), providing an outlook on future stock prices.
- **Visualization**: The `visualize_forecast()` method generates visualizations comparing historical stock prices with the forecasted data from the selected model.
- **Trend & Volatility Analysis**: The `analyze_forecast()` method evaluates the forecast for potential trends (upward, downward, or stable) and volatility by analyzing the forecasted confidence intervals.

#### How to Use:
1. **Class Initialization**: Instantiate the `ForecastFutureMarkets` class by passing the stock ticker and the path to the processed stock data file.
    ```python
    forecaster = ForecastFutureMarkets(ticker="AAPL", processed_file_path="processed_data.csv")
    ```
2. **Model Loading**: Load all necessary models using the `load_all_models()` method.
    ```python
    forecaster.load_all_models()
    ```
3. **Forecasting**:
    - ARIMA Forecast:
        ```python
        forecast_arima = forecaster.forecast_arima(steps=180)
        ```
    - SARIMA Forecast:
        ```python
        forecast_sarima, conf_int = forecaster.forecast_sarima(steps=180)
        ```
    - LSTM Forecast:
        ```python
        forecast_lstm = forecaster.forecast_lstm(steps=180)
        ```
4. **Visualization**: Use the `visualize_forecast()` method to visualize the forecast.
    ```python
    forecaster.visualize_forecast(forecast_arima, model_name="ARIMA")
    ```
5. **Trend and Volatility Analysis**: Analyze the forecast using `analyze_forecast()`.
    ```python
    forecaster.analyze_forecast(forecast_arima)
    ```

For complete end-to-end execution, the forecasting process is integrated into the `main/main.py` script, which automates loading models, generating forecasts, and running the analysis pipeline.

---


### 4. Portfolio Optimization

This section of the project focuses on optimizing a portfolio consisting of three assets (TSLA, BND, SPY) using forecasts generated by an ARIMA statistical model. The model predicts the price trends for the next 6-12 months, and these predictions are then used for portfolio optimization. The portfolio optimization process involves the following steps:

#### 4.1. Forecasting Future Market Trends

The portfolio optimization is based on forecasted returns for the assets. The forecasting is done using the **ARIMA (AutoRegressive Integrated Moving Average)** model, which is applied to historical data for each asset. This model is used to predict future market trends for each asset, providing the expected returns over the forecast horizon.

1. **Data Preprocessing**: Historical stock data is preprocessed, ensuring that it is cleaned and ready for ARIMA forecasting.
2. **ARIMA Model**: The ARIMA model is trained on each asset's historical price data and used to generate forecasts for the next 6-12 months.
3. **Forecasting**: The generated forecasts are then stored and merged for all three assets (TSLA, BND, SPY).

#### 4.2. Portfolio Performance Calculation

The portfolio's performance is evaluated based on the forecasted expected returns and volatility. The main performance metrics calculated are:

- **Annual Return**: The expected return for each asset is annualized, assuming 252 trading days per year.
- **Covariance Matrix**: A covariance matrix of returns is calculated for the three assets, which captures the relationship between their returns.
- **Portfolio Return & Volatility**: The weighted returns and volatility of the portfolio are calculated using the forecasted data and the covariance matrix.

#### 4.3. Portfolio Optimization

Portfolio optimization aims to maximize the Sharpe ratio by adjusting the portfolio weights. The optimization process involves:

1. **Objective Function**: The negative Sharpe ratio is minimized, which involves maximizing the portfolio's return for a given level of risk (volatility).
2. **Constraints**: The portfolio weights are constrained such that the sum of weights equals 1, and individual weights are between 0 and 1.
3. **Optimization**: The optimization is done using the `SLSQP` (Sequential Least Squares Quadratic Programming) method from `scipy.optimize.minimize`.

#### 4.4. Visualizing Portfolio Performance

Once the optimal portfolio weights are found, the performance of the portfolio is visualized by simulating its growth over time. This is done by:

- Calculating the cumulative return of the portfolio based on the optimal weights.
- Plotting the portfolio's growth and showing the cumulative return over time.

Additionally, a heatmap of the covariance matrix is generated to visualize the relationships between the assets and their return correlations.

#### 4.5. Risk Management - Value at Risk (VaR)

To measure the potential risk of the portfolio, the **Value at Risk (VaR)** is calculated at a 95% confidence level. This gives an estimate of the potential loss in the portfolio under normal market conditions.

### Example Usage

```python
Open the `portfolio_optimizer.iypnb` notebook and run each process
You can see the intermediate results while the main pipeliner for the whole project process is executed by running `python main/main.py`

```

---



## Future Enhancements

- Fine-tune the parameters of the LSTM model to reduce the mae, mape, and rmse
- Differenciate the price data as it is a non-stationary data and analyze based on the differenciated dat


----



### More information
- You can refer to [this link]() to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/portfolio-management-optimization/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
