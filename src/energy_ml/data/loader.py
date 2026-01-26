import pandas as pd

def load_energy_data(path: str) -> pd.DataFrame:
    """
    Load and return raw energy consumption data.
    """
    df = pd.read_csv(path)
    return df