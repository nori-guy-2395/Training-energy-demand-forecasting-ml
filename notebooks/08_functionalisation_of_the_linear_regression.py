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
import energy_ml.analysis.plotting as pltting



Data_frame = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
df, issues = loader.Data_formatting_clean_reassemble(Data_frame)
print(df.info())

Data_fitting = 'Voltage'
Features = ['hour', 'dayofweek']

model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = Analysis.Linear_regression_function_time(Features, df, Data_fitting)

start_time = pd.Timestamp("2007-01-01 00:00")
end_time   = pd.Timestamp("2007-03-01 00:00")

# start_time = False
# end_time   = pd.Timestamp("2007-03-01 00:00")

# start_time = pd.Timestamp("2007-03-01 00:00")
# # end_time   = False

pltting.plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time, end_time )




