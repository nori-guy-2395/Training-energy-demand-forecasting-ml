import matplotlib.pyplot as plt

import pandas as pd

class TimeSeriesStats:
    def __init__(self, df, time_col, value_col):
        self.df = df.copy()
        self.time_col = time_col
        self.value_col = value_col
        self.df[time_col] = pd.to_datetime(self.df[time_col])

    def _filter_by_time(self, start=None, end=None):
        df = self.df
        if start:
            df = df[df[self.time_col] >= pd.to_datetime(start)]
        if end:
            df = df[df[self.time_col] <= pd.to_datetime(end)]
        return df

    def describe(self, start=None, end=None):
        df = self._filter_by_time(start, end)
        series = df[self.value_col]

        stats = {
            "mean": series.mean(),
            "median": series.median(),
            "mode": series.mode().iloc[0] if not series.mode().empty else None,
            "min": series.min(),
            "max": series.max(),
            "count": series.count()
        }

        return pd.DataFrame(stats, index=["value"])
    

def Time_series_analysis(Date_data, start = None, end = None) :   
    
    if start:
        Date_data = Date_data >= pd.to_datetime(start)
    if end:
        Date_data = Date_data <= pd.to_datetime(end)
    
    return Date_data

def Describe_the_data(Date_data, Data_analysis):
    
    
    
    return