# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:03:25 2026

@author: gregr
"""

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




Data_frame = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
df, issues = loader.Data_formatting_clean_reassemble(Data_frame)
print(df.info())

Data_fitting = 'Voltage'
Features = ['hour', 'dayofweek']

model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = Analysis.Linear_regression_function_time(Features, df, Data_fitting)






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



# Define start and end timestamps so that we can get a zoomed in look
start_time = pd.Timestamp("2007-01-01 00:00")
end_time   = pd.Timestamp("2007-03-01 00:00")

# Filter train and test DataFrames zooming
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