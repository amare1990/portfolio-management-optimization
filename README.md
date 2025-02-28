# Portfolio Management Optimization

> Portfolio Management Optimization is a data science project aimed at analyzing and optimizing a financial portfolio using historical stock market data. It enables users to download and preprocess stock data from Yahoo Finance, perform exploratory data analysis (EDA) to gain insights, conduct volatility analysis and outlier detection, compute financial metrics like Value at Risk (VaR) and Sharpe Ratio, and summarize key insights and trends. This project is implemented in Python and leverages powerful financial libraries.

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
│   ├── plots/eda                      # Directory for storing generated plots while performing EDA
│       ├── closing_prie.png    # Time series plot
│       ├── daily_pct_change.png
│
│── scripts/                        # Python scripts for different modules
    ├── __init__.py
│   ├── data_analysis.py            # Data downloading, cleaning, analysis, normalization, and EDA
│
│── src/                            # Main automation script
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
  python src/src.py
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

- Check code formatting in the `src/` directory (main processing pipeline):
  ```sh
  pylint src/src.py
  ```

- Auto-fix linter errors in the `src/` directory:
  ```sh
  autopep8 --in-place --aggressive --aggressive src/src.py
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
python src/src.py

```

#### 2.8 Expected Outputs
- Processed CSV files with cleaned and normalized stock data.
- Visualizations of price trends, volatility, and outliers.
- Computed VaR and Sharpe Ratio for risk assessment.

## Future Enhancements
- Forecast using statistical models and LSTM
- Forecast Future Market Trends using forecasted data and built models
- Optimize Portfolio Based on Forecast data






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
