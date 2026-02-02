# src/data_loader.py

import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
from config import OBS_DATA_DIR, MODEL_DATA_DIR

class ObsLoader:
    """
     Initialize the observation data loader.

     Args:
        data_dir: Directory containing observation files.
        
        Uses OBS_DATA_DIR from config if None.
    """

    ERR_AOD = 0.021
    REFERENCE_TIME = dt.datetime(2022, 12, 31, 1, 0, 0)
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir if data_dir is not None else OBS_DATA_DIR
        self.data = None
        self.raw_data = None

    def load_single_year(self, filename: str) -> pd.DataFrame:
        """
        Load observation data for a single year/file.

        CSV file
        
        Args:
            filename: Name of the observation file to load
            
        Returns:
            DataFrame with raw observation data
        """

        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        header_row = 0
        with open(filepath, 'r', encoding='latin1') as f:
            for i, line in enumerate(f):
                if "Date" in line:
                    header_row = i
                    break
        
        data = pd.read_csv(filepath, skiprows=header_row, delimiter=',', encoding= 'latin1')
        data.reset_index(inplace=True, drop=True)
        return data
    
    def load_multiple_year(self, filenames: list[str]) -> pd.DataFrame:
        """
        Load and concatenate observation data from multiple years.
        
        Args:
            filenames: List of observation filenames to load and concatenate
            
        Returns:
            Concatenated DataFrame with all years
        """
        datas = []
        index_list = {}
        for filename in filenames: 
            df = self.load_single_year(filename)
            idx = df.iloc[-1]
            index_list[f"{filename}"] = idx
            datas.append(df)
        
        self.raw_data = pd.concat(datas, axes=0, ignore_index=True)
        return self.raw_data, index_list

    def select_process_obs(self):
        """
        Process raw observation data: decode time, calculate AOD550, filter invalid values.
        
        Args:
            year_indices: List of last indices for each year (for time adjustment across years)
            
        Returns:
            Processed DataFrame ready for analysis
        """

        if self.raw_data is None:
            raise ValueError("No data loaded. Use load_single_year() or load_multiple_years() first.")

        
        

    def _add_550nm(self): 
        """
        Use the Beer-Bouguer-Lambert law to have the AOD at 550 nm to
        
        have the same wavelength to compare with the model.
        """

        if self.data is None: 
            raise ValueError("Need to load data first, use load() method")
        
        #print(f"Use key words of the dataframe: {self.data.head()}")
        err_ang = abs(-(1/np.log(500/675))*(ERR_AOD/self.data["AOD_500nm"])-(ERR_AOD/self.data["AOD_675nm"]))
        aod550 = self.data["AOD_500nm"]*(550/500)**(-self.data["440-870_Angstrom_Exponent"])
        err_aod550 = (ERR_AOD*(aod550/self.data["AOD_550nm"]) + err_ang*abs(np.log(500/550))*aod550)
        self.data = pd.concat([self.data, pd.concat([aod550, err_aod550], axis=1)], axis=1)
    
    def get_date_range(self, start: str, end: str) -> pd.DataFrame:
        """
        Filters observations based on the required date range of observations.
        """

        if self.data is None: 
            raise ValueError("Need to load data first, use load() method")
        else: 
            mask = (self.data['time'] >= start) & (self.date['time'] <= end)
        
        return self.data[mask]
    
class ModLoader:
    """
    Loads and manages simulated data by the MINNI-FORAIR-IT model.

    NetCDF4 file
    """

    REFERENCE_TIME = dt.datetime(1900, 1, 1, 0, 0, 0)
    RAYLEIGH_AOD = 0.0729

    def __init__(self, data_dir=None):
        self.data_dir = data_dir if data_dir is not None else MODEL_DATA_DIR
        self.data = None
    
    def load(self, filename: str) -> xr.Dataset:
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.dataset = xr.open_dataset(filepath)
        return self.dataset
    
    def get_variable(self, var: str) -> xr.DataArray:
        """ Extract a specific variable """

        if self.dataset is None: 
            raise ValueError("Need to load data first, use load()")
        else:
            if var not in self.dataset: 
                raise KeyError(f"Variable {var} not found. {self.dataset.variables}")
            
        return self.dataset[var]
    
