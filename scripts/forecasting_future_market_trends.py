"""USing the already trained price forecasting model."""

import numpy as np
import pandas as pd


class Forecast_Future_Markets:
  def __init__(self, processed_file_path):
    self.data = pd.read_csv(processed_file_path)



