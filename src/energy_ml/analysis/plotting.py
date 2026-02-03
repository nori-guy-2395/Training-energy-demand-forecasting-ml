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
    
    
    
    
def plotting_model_full_scope(Data_fitting,t_train,y_train,y_train_pred, t_test,y_test,y_test_pred):
    '''
    Parameters
    
    Output from the trained version of the model to plot the trained and test data
    also gives original data through combination of the the train and test data (concaternated)
    
    Returns
    null: plots the figures directly.
    
    '''
        
    train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
    test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})
    
    plt.figure(figsize=(14,6))
    plt.plot(pd.concat([train_df, test_df]).sort_values("Datetime")["Datetime"],
             pd.concat([train_df, test_df]).sort_values("Datetime")["Actual"],
             label="Actual", color="black", alpha=0.6)
    plt.scatter(train_df["Datetime"], train_df["Predicted"], label="Train Prediction", color="blue", s=10)
    plt.scatter(test_df["Datetime"], test_df["Predicted"], label="Test Prediction", color="red", s=10)
    plt.xlabel("Datetime")
    plt.ylabel(Data_fitting)
    plt.title("Actual vs Predicted Energy Consumption with feature(s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    return    
    
def plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time = None, end_time = None):
    
    '''
    Parameters
    
    Output from the trained version of the model to plot the trained and test data
    also gives original data through combination of the the train and test data (concaternated)
    
    Returns
    null: plots the figures directly.
    
    '''
    
    
    
    # def plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time=None, end_time=None):

    train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
    test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})

    # Find global bounds
    global_start = min(train_df["Datetime"].min(), test_df["Datetime"].min())
    global_end   = max(train_df["Datetime"].max(), test_df["Datetime"].max())

    # Fill missing inputs
    if start_time is None:
        start_time = global_start

    if end_time is None:
        end_time = global_end

    print("Using start_time:", start_time)
    print("Using end_time:", end_time)

    # Filter once
    train_zoom = train_df[(train_df["Datetime"] >= start_time) & (train_df["Datetime"] <= end_time)]
    test_zoom  = test_df[(test_df["Datetime"] >= start_time) & (test_df["Datetime"] <= end_time)]
    


    fig, ax = plt.subplots(figsize=(12,5))

    # Make pretty

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # Actual
    ax.plot(pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Datetime"],
             pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Actual"],
             label="Actual", color="black", alpha=0.6)

    # Predictions
    ax.scatter(train_zoom["Datetime"], train_zoom["Predicted"], label="Train Prediction", color="blue", s=10)
    ax.scatter(test_zoom["Datetime"], test_zoom["Predicted"], label="Test Prediction", color="red", s=10)

    ax.set_xlabel("Datetime")
    ax.set_ylabel(Data_fitting)
    # ax.set_title("Zoomed Energy Consumption with" + str(Features) + "feature(s)")
    ax.set_title("Zoomed Energy Consumption with feature(s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    return    
    