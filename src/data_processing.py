# src/data_processing.py

import pandas as pd
import xarray as xr 

class DataCleanser:
    """Manipulate and handle data with averages, error propagation and filters"""

    def __init__(self, data) -> pd.DataFrame:
        self.data = data

    def clean_data(self):

        """Clean data from values like -999 and <= 0"""

        if self.data is None:
            raise ValueError("Need to load data first, use load()")
        else: 
            mask = (self.data)