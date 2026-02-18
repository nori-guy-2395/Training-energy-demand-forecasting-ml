''' 

Grid search of the parameters for the random forest regression

Need to check the feature importance

Want to explore additional values, relationship to data within the data (voltage vs Global_power)

Can the splitting of the data be, instead of randomly split over the whole dataset, be early part -> later part 

ie. training built on first 80% of data test on the last 20% 


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
from sklearn.ensemble import RandomForestRegressor


Data_frame = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
df, issues = loader.Data_formatting_clean_reassemble(Data_frame)
print(df.info())

Data_fitting = 'Global_active_power'


Features = ['rolling_24_std','rolling_24_mean','lag_1','lag_24', 'hour', 'dayofweek', 'month', 'day']

# Features = ['rolling_24_std','rolling_24_mean','lag_1','lag_24']

'''

Trying a top and tail split for the test and training. This is closer to reality/useage. We can't see the future.

Should be able to continuously train on the data as it come thrugh... Ie increase the train data gradually for optimisation? 



'''

# model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = Analysis.Random_forest_regression_function_start_End_split(Features, df, Data_fitting)


# start_time = pd.Timestamp("2010-02-01 00:00")
# end_time   = pd.Timestamp("2010-02-10 00:00")

# pltting.plotting_model_full_scope(Data_fitting, t_train,y_train,y_train_pred, t_test,y_test,y_test_pred)


# pltting.plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time, end_time)

# Analysis.analyse_the_best_features_random_forest(Features, df, Data_fitting)



'''

My function writing could be improved. I think that the functions seem to be typically one use and unique. Maybe splitting them up is a 

good thing to do. Maybe also clean up the interface? Would like a menu of fitting and data to analyse... Could be good/multipuprose for new data downloads

Want to learn some query functions as well. Is python the best for this... ???

Adding a slight shift to the voltage_01 0.1 shift try to predict the Global_acive_power from the voltage.


'''
Features = ['rolling_24_std','rolling_24_mean','lag_1','lag_24', 'hour', 'dayofweek', 'month', 'day', 'Voltage_01']

Analysis.analyse_the_best_features_random_forest(Features, df, Data_fitting)