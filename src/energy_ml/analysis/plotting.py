import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def create_basic_figure_of_data(x, y, x_label, y_label, Date=False):
    
    """
    Standard input of plotting function 
    
    no return but does plot a figure of the data assuming x is a datetime value
    
    """
    
    fig, ax = plt.subplots()

    ax.plot(x,y)
    
    # ax.set_ylim(y_range)
    # ax.set_xlim(x_range)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)    
    
    
    if Date == True:
        
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")

        # Use AutoDateLocator and AutoDateFormatter
        # locator = mdates.AutoDateLocator()      # automatically choose tick positions
        # formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
    plt.show()
    
    
    return


def create_basic_figure_for_each_column_against_datetime(Data_frame):
    
    for col in Data_frame.columns:
        series = Data_frame[col]
        
        if not col.lower().startswith("Datetime"):
            create_basic_figure_of_data(Data_frame['Datetime'], series, 'Date and time [days months, hours seconds]', col, Date=True)
    
    return



def figure_plotting_time_intervanl2(Data_frame, datetime_col, value_col, x_label, y_label, Date = True, start = None,  end =None):
    
    
    """
    Standard input of plotting function plus a time interval defined, or not as default
    
    no return but does plot a figure of the data assuming x is a datetime value
    
    determine for a datetime range given here.

    start/end can be:
    - '2010'
    - '2010-11'
    - '2010-11-26'
    """

    Data_frame = Data_frame.copy()
    Data_frame[datetime_col] = pd.to_datetime(Data_frame[datetime_col])
    Data_frame = Data_frame.set_index(datetime_col).sort_index()

    if end is None:
        y_data = Data_frame.loc[start, value_col]
        x_data = Data_frame.loc[start, datetime_col]
    else:
        y_data = Data_frame.loc[start:end, value_col]
        x_data = Data_frame.loc[start, datetime_col]

    if y_data.empty:
        raise ValueError("No data found for the given time range")
    
    x = x_data
    y = y_data
    
    fig, ax = plt.subplots()

    ax.plot(x,y)
    
    # ax.set_ylim(y_range)
    # ax.set_xlim(x_range)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)    
    
    
    if Date == True:
        
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")

        # Use AutoDateLocator and AutoDateFormatter
        # locator = mdates.AutoDateLocator()      # automatically choose tick positions
        # formatter = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
    plt.show()
    
    
    return


def figure_plotting_time_interval(
    Data_frame,
    datetime_col,
    value_col,
    x_label,
    y_label,
    Date=True,
    start=None,
    end=None
):
    """
    Plot data over a specified datetime interval.

    start/end can be:
    - '2010'
    - '2010-11'
    - '2010-11-26'
    """

    Data_frame = Data_frame.copy()
    Data_frame[datetime_col] = pd.to_datetime(Data_frame[datetime_col])
    Data_frame = Data_frame.set_index(datetime_col).sort_index()

    # Slice data
    if start is None and end is None:
        y_data = Data_frame[value_col]
    elif end is None:
        y_data = Data_frame.loc[start, value_col]
    else:
        y_data = Data_frame.loc[start:end, value_col]

    if y_data.empty:
        raise ValueError("No data found for the given time range")

    # x-axis comes from the index
    x = y_data.index
    y = y_data.values

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if Date:
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

    plt.show()