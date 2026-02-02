# test_obs_loader.py

import pandas as pd
from pathlib import Path
from src.data_loader import ObsLoader

def test_load_csv():
    loader = ObsLoader(data_dir=Path("Lampedusa_obs"))
    df = loader.load("20230101_20231232_Lampedusa.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
