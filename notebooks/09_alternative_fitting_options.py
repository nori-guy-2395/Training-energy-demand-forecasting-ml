'''

Implement new linear regression features

Want to include rolling and lag based fitting. For daily/hourly cycles.

Should fit for the recent past and continuously assess. 


Lag features are what has happened recently in the past, asking what were the properties eg. 1 hour ago

Rolling features are what the statistical averages over th recent past eg. average or std 

Reload the data

Bring the Analysis script with the linear regression feature function back

Add new feature analysis for the lag and rolling features

Check the new plots. 

Use recent plotting zoom function for looking closely at the data

(should also look at appending data for easier data handling in for test?)


'''

import numpy as np
import energy_ml.data.loader as loader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import energy_ml.analysis.Basic_stats_scripts as Analysis
import energy_ml.analysis.plotting as pltting


Data_frame = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
df, issues = loader.Data_formatting_clean_reassemble(Data_frame)
print(df.info())

Data_fitting = 'Global_intensity'
# Features = ['hour', 'dayofweek']

        # if i == 'hour':
        #     df[i] = df["Datetime"].dt.hour
        # if i == 'dayofweek':
        #     df[i] = df["Datetime"].dt.dayofweek
        # if i == 'month':
        #     df[i] = df["Datetime"].dt.month
        # if i == 'day':
        #     df[i] = df["Datetime"].dt.day
        # if i == 'year':
        #     df[i] = df["Datetime"].dt.year
        # if i == 'quarter':
        #     df[i] = df["Datetime"].dt.quarter


'added the rolling features and the lag'
Features = ['rolling_24_std','rolling_24_mean','lag_1','lag_24', 'hour', 'dayofweek', 'month', 'day', 'year', 'quarter']

model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = Analysis.Linear_regression_function_time(Features, df, Data_fitting)


start_time = pd.Timestamp("2007-01-01 00:00")
end_time   = pd.Timestamp("2007-01-05 00:00")



pltting.plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time, end_time )



'''

The data fitting is improved massively

the local treatment of the data over the rolling mean/std is very good

Analysis of the best combination can be carried out

Automate the process? 


'''


from itertools import combinations
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Features = [
#     'rolling_24_std','rolling_24_mean','lag_1','lag_24',
#     'year','hour','dayofweek','month','day','quarter'
# ]

'''

loop over the number of features that are in the list

'''
total = sum(1 for _ in range(1, len(Features) + 1) for __ in combinations(Features, _)) # Set up the Features and the range. 
i = 0
results = []

for r in range(1, len(Features) + 1): # Loop start
    for combo in combinations(Features, r):  # Combo features
        i += 1
        print(f"{i}/{total} -> {combo}")
        model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = \
            Analysis.Linear_regression_function_time(list(combo), df, Data_fitting)  # Use the previous function to fit to the data and analyse

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

import matplotlib.pyplot as plt

mae_summary = (
    df_results
    .groupby("n_features")[["train_mae", "test_mae"]]
    .mean()
    .reset_index()
) # Save and extract the data 

plt.figure(figsize=(8, 5))
plt.plot(mae_summary["n_features"], mae_summary["train_mae"], marker="o", label="Train MAE")
plt.plot(mae_summary["n_features"], mae_summary["test_mae"], marker="o", label="Test MAE")
plt.xlabel("Number of Features")
plt.ylabel("MAE")
plt.title("MAE vs Number of Features")
plt.legend()
plt.grid(True)
plt.show()

df_results["mae_gap"] = df_results["test_mae"] - df_results["train_mae"]   # which is the overfitted data


df_overfit = df_results.sort_values("mae_gap", ascending=False) # Sort the data

print(df_overfit.head(10)[["features", "n_features", "train_mae", "test_mae", "mae_gap"]])  


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




