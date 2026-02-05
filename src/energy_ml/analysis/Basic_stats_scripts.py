import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations




import ast

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
        if i == 'lag_1':
            df[i] = df[Fitting_to].shift(1)            
        if i == 'lag_24':
            df[i] = df[Fitting_to].shift(24)
        if i == 'rolling_24_mean':
            df[i] = df[Fitting_to].shift(1).rolling(window=24).mean()
        if i == 'rolling_24_std':
            df[i] = df[Fitting_to].shift(1).rolling(window=24).std()
        else:
            print("problem with the called data. Try: hour, dayofweek, month, day, year, quarter, lag_1, lag_24, rolling_24_mean, rolling_24_std")

    valid_mask = df[Fitting_to].notna()
    df_valid = df[valid_mask].copy()  # This says where there are values that are not valid for cropping out

    # print('check', type(df_valid), type(valid_mask))

    # print('feature', Feature)    
 
    # Feature_dt = Feature + ['Datetime']
    
    df_valid = df.dropna(subset=Feature + [Fitting_to]).copy()
    
    # print('feature', Feature)
    # 
    Feature_dt = Feature + ['Datetime']
    
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

    print('function print, the metric:', metrics)


    train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
    test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})


    
    return model, metrics, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred

def load_and_analyse_results(csv_path, overfit_quantile=0.9):
    """
    Loads saved feature search results and:
    
        
    Plots MAE vs number of features (train vs test)
    
    Detect any overfitting 
    
    Compute & plot Pareto-optimal models
    """

    # Load the results
    df_results = pd.read_csv(csv_path)

    # Check and parse the features such taht the load is safely handled.
    if isinstance(df_results.loc[0, "features"], str):
        df_results["features"] = df_results["features"].apply(ast.literal_eval)

    # MAE comparisons
    mae_summary = (
        df_results
        .groupby("n_features")[["train_mae", "test_mae"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(mae_summary["n_features"], mae_summary["train_mae"], marker="o", label="Train MAE")
    plt.plot(mae_summary["n_features"], mae_summary["test_mae"], marker="o", label="Test MAE")
    plt.xlabel("Number of Features")
    plt.ylabel("MAE")
    plt.title("MAE vs Number of Features")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Checkking for overfitting. OVerfitting above gap between test and train
    df_results["mae_gap"] = df_results["test_mae"] - df_results["train_mae"]
    threshold = df_results["mae_gap"].quantile(overfit_quantile)
    df_results["overfit_flag"] = df_results["mae_gap"] > threshold

    print("\n Highest overfitted models:")
    print(df_results.sort_values("mae_gap", ascending=False)
          .head(10)[["features", "n_features", "train_mae", "test_mae", "mae_gap"]])

    # Pareto sorting
    pareto_rows = []
    for i, row in df_results.iterrows():
        dominated = False
        for j, other in df_results.iterrows():
            if (
                (other["test_mae"] <= row["test_mae"]) and
                (other["n_features"] <= row["n_features"]) and
                (
                    (other["test_mae"] < row["test_mae"]) or
                    (other["n_features"] < row["n_features"])
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto_rows.append(row)

    df_pareto = pd.DataFrame(pareto_rows).sort_values(["n_features", "test_mae"])

    print("\n  Pareto-optimal models (lowest features highest reliability):")
    print(df_pareto[["features", "n_features", "test_mae", "test_rmse"]])

    # plotting the paretto figure
    plt.figure(figsize=(8, 5))
    plt.scatter(df_results["n_features"], df_results["test_mae"], alpha=0.3, label="All models")
    plt.scatter(df_pareto["n_features"], df_pareto["test_mae"], label="Pareto-optimal")
    plt.xlabel("Number of Features")
    plt.ylabel("Test MAE")
    plt.title("Pareto Front: Model Complexity vs Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_results, df_pareto

def Random_forest_regression_function(Feature, df, Fitting_to):

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
        if i == 'lag_1':
            df[i] = df[Fitting_to].shift(1)            
        if i == 'lag_24':
            df[i] = df[Fitting_to].shift(24)
        if i == 'rolling_24_mean':
            df[i] = df[Fitting_to].shift(1).rolling(window=24).mean()
        if i == 'rolling_24_std':
            df[i] = df[Fitting_to].shift(1).rolling(window=24).std()
        else:
            print("problem with the called data. Try: hour, dayofweek, month, day, year, quarter, lag_1, lag_24, rolling_24_mean, rolling_24_std")

    valid_mask = df[Fitting_to].notna()
    df_valid = df[valid_mask].copy()  # This says where there are values that are not valid for cropping out

    # print('check', type(df_valid), type(valid_mask))

    # print('feature', Feature)    
 
    # Feature_dt = Feature + ['Datetime']
    
    df_valid = df.dropna(subset=Feature + [Fitting_to]).copy()
    
    # print('feature', Feature)
    # 
    Feature_dt = Feature + ['Datetime']
    
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
####  SWAPPED OUT THE LINEAR REGRESSION MODEL FOR A RANDOM FOREST REGRESSOR  - "STANDARD SETTINGS" !
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
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

    print('function print, the metric:', metrics)


    train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
    test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})


    
    return model, metrics, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred



def analyse_the_best_features_random_forest(Features, df, Data_fitting):
   
    total = sum(1 for _ in range(1, len(Features) + 1) for __ in combinations(Features, _)) # Set up the Features and the range. 
    i = 0
    results = []

    for r in range(1, len(Features) + 1): # Loop start
        for combo in combinations(Features, r):  # Combo features
            i += 1
            print(f"{i}/{total} -> {combo}")
            model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = \
                Random_forest_regression_function(list(combo), df, Data_fitting)  # Use the previous function to fit to the data and analyse

            test_mae = metric['Test MAE']
            test_rmse = metric['Test RMSE']

            train_mae = metric['Train MAE']
            train_rmse = metric['Train RMSE']

            results.append({
                "features": combo,
                "n_features": len(combo),
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse
            }) # Create and add data to  dictionary of the results

    df_results = pd.DataFrame(results) # Convert to pandas frame

    df_ranked = df_results.sort_values(
        by=["test_mae", "test_rmse"],
        ascending=[True, True]
    ) # Sort and nalyse the data 

    print(df_ranked.head(10)) # Show which are the highest values

    best_row = df_ranked.iloc[0] # Which is the best feature
    best_features = list(best_row["features"])

    print("Best features:", best_features) 
    print("Test MAE:", best_row["test_mae"]) # Show!
    print("Test RMSE:", best_row["test_rmse"])

    df_ranked.to_csv(Data_fitting+"linear_regression_feature_search.csv", index=False) # save data
    
    
    pareto_rows = [] #Empty list
    
    for i, row in df_results.iterrows(): # Loop over all results
        dominated = False  # Lets assume that the best is no found yet
        for j, other in df_results.iterrows():
            if (
                (other["test_mae"] <= row["test_mae"]) and
                (other["n_features"] <= row["n_features"]) and
                (
                    (other["test_mae"] < row["test_mae"]) or
                    (other["n_features"] < row["n_features"])
                )
            ):  # Check, is this the simplest model or the best model. If best then dominant, if simplest then dominant. 
                dominated = True
                break
        if not dominated:
            pareto_rows.append(row) # Edge case. 
    
    df_pareto = pd.DataFrame(pareto_rows).sort_values(["n_features", "test_mae"]) # plot the best features in the pareto plots. 
    print(df_pareto[["features", "n_features", "test_mae"]]) 
    
    plt.figure(figsize=(8, 5))
    plt.scatter(df_results["n_features"], df_results["test_mae"], alpha=0.3, label="All models")
    plt.scatter(df_pareto["n_features"], df_pareto["test_mae"], color="red", label="Pareto-optimal")
    plt.xlabel("Number of Features")
    plt.ylabel("Test MAE")
    plt.title("Pareto Front: Model Complexity vs Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    return 
























