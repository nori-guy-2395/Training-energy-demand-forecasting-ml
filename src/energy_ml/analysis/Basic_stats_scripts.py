import matplotlib.pyplot as plt
import pandas as pd


def Information_data_analysis(Data_frame, Data_header, Time_frame_analysis = 'Y', show_mean = False):


    Data_frame.set_index('datetime', inplace=True)
    
    # if show_mean == True:
    #     # Daily mean (already daily here, but useful for longer data)
    #     weekly_mean = Data_frame[Data_header].resample('W').mean()
    #     print('Weekly mean: ', weekly_mean)
    
    # # Weekly statistics
    # Yearly_stats = Data_frame[Data_header].resample(Time_frame_analysis).agg(['mean', 'median', 'min', 'max', 'std'])
    # print(Yearly_stats)

    return



def time_period_stats(df, datetime_col, value_col, start, end=None):
    """
    Calculate statistics for a datetime range.

    start/end can be:
    - '2010'
    - '2010-11'
    - '2010-11-26'
    """

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    if end is None:
        data = df.loc[start, value_col]
    else:
        data = df.loc[start:end, value_col]

    if data.empty:
        raise ValueError("No data found for the given time range")

    return {
        "mean": data.mean(),
        "median": data.median(),
        "mode": data.mode().tolist(),
        "min": data.min(),
        "max": data.max(),
        "std": data.std()
    }