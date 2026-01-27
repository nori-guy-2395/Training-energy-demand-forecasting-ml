import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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

def mean_of_data(Data, start = None, end = None):

    mean = np.mean(Data[start:end])
    
    return mean


def create_basic_figure_for_each_column_against_datetime(Data_frame):
    
    for col in Data_frame.columns:
        series = Data_frame[col]
        
        if not col.lower().startswith("Datetime"):
            create_basic_figure_of_data(Data_frame['Datetime'], series, 'Date and time [days months, hours seconds]', col, Date=True)
    
    return