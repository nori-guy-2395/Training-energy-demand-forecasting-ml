import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

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


def Linear_regression_function_time(Feature, df, Fitting_to):

    '''
    Linear regression fitting to dataset
    
    Define features in time variables automatically  chooses the datatime64 pandas frame
    
    Parameters
    ----------
    Feature 
    list of features to be used to produce linear regression fit
    
    df
    Panads data frame of the data to take a fit
    
    MUST have Datetime collumn for this function to work 
    

    Returns     
    ----------
    Model
    Trained linear model object
    
    Metrics
    RSME and MAE defaulat returns from the linear regression model
    
    Percentage metrics
    Normalised RSME and MAE from the data set 
   

    '''
    # # Assume df_features has 'Datetime' and 'Global_active_power'
    df = df.copy()
    
    # # features that are then enurmerated so that the date/days/months can be enumerated
    # df["hour"] = df["Datetime"].dt.hour
    # df["dayofweek"] = df["Datetime"].dt.dayofweek
    # df["month"] = df["Datetime"].dt.month
    # df["day"] = df["Datetime"].dt.day
    # df["year"] = df["Datetime"].dt.year
    # df["quarter"] = df["Datetime"].dt.quarter

    for i in Feature:
        if i == 'hour':
            df[i] = df["Datetime"].dt.hour
        if i == 'dayofweek':
            df[i] = df["Datetime"].dt.dayofweek
        if i == 'month':
            df[i] = df["Datetime"].dt.month
        if i == 'day':
            df[i] = df["Datetime"].dt.day
        if i == 'year':
            df[i] = df["Datetime"].dt.year
        if i == 'quarter':
            df[i] = df["Datetime"].dt.quarter
        else:
            print("problem with the called data. Try: hour, dayofweek, month, day, year, quarter")

    valid_mask = df[Fitting_to].notna()
    df_valid = df[valid_mask].copy()  # This says where there are values that are not valid for cropping out

    print('check', type(df_valid), type(valid_mask))

    print('feature', Feature)    

    Feature_dt = Feature + ['Datetime']

    print('feature', Feature)

    X = df_valid[Feature_dt].copy()  # keep Datetime for plotting
    y = df_valid[Fitting_to] # This is the fitting value in y


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) # Splitting the data

    # Save timestamps for plotting later
    t_train = X_train["Datetime"]
    t_test = X_test["Datetime"]

    # Drop Datetime for model input as the time data would be super bad (can't fit to datatime64 format)
    X_train_model = X_train[Feature]
    X_test_model  = X_test[Feature]


    #   Linear regression !!!!
    model = LinearRegression()
    model.fit(X_train_model, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_model)
    y_test_pred  = model.predict(X_test_model)

    # Metrics --- Currently no analysis or discussion of the data
    metrics = {
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

    print(metrics)


    train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
    test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})


    
    return model, metrics, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred