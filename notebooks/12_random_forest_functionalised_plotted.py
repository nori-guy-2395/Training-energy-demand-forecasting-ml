'''

Functionalise the random forest model

Load, run model, plot, analysis and then loop the analysis to find the best pareto-optimisation situation. 

Not sre all options have been explored in the loop for pareto optimisation !!!! 

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

Data_fitting = 'Global_active_'
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

model, metric, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred = Analysis.Random_forest_regression_function(Features, df, Data_fitting)


start_time = pd.Timestamp("2007-01-01 00:00")
end_time   = pd.Timestamp("2007-01-05 00:00")



pltting.plotting_model_zoom(Data_fitting, t_train, t_test, y_train, y_test, y_train_pred, y_test_pred, start_time, end_time )

Analysis.analyse_the_best_features_random_forest(Features, df, Data_fitting)

'''

                                   features  n_features  ...  test_mae   mae_gap
166                (dayofweek, month, year)           3  ...  3.390361  0.004807
380           (dayofweek, month, day, year)           4  ...  3.390355  0.004804
170              (dayofweek, year, quarter)           3  ...  3.388185  0.004796
383         (dayofweek, day, year, quarter)           4  ...  3.388177  0.004793
382       (dayofweek, month, year, quarter)           4  ...  3.385825  0.004707
636  (dayofweek, month, day, year, quarter)           5  ...  3.385817  0.004704
47                        (dayofweek, year)           2  ...  3.394850  0.004624
168                  (dayofweek, day, year)           3  ...  3.394850  0.004624
45                       (dayofweek, month)           2  ...  3.390495  0.004576
48                     (dayofweek, quarter)           2  ...  3.388274  0.004568

[10 rows x 5 columns]
                                          features  n_features  test_mae
2                                         (lag_1,)           1  0.423912
19                        (rolling_24_mean, lag_1)           2  0.420833
55        (rolling_24_std, rolling_24_mean, lag_1)           3  0.418596
179  (rolling_24_std, rolling_24_mean, lag_1, day)           4  0.418596


best is consistent with the linear regression, where rolling means and std, with lag appears to be best. 

The improvement with the addition of the rolling and stds on the lag isapparement for the test MAE. 

The addition of the day appears to do nothing to the test MAE to 6 sigfigs. 

'''