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

import pandas as pd

df_results = pd.read_csv("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/linear_regression_feature_search.csv")

# # If features were saved as tuples -> convert back safely
df_results["features"] = df_results["features"].apply(lambda x: tuple(x.strip("()").replace("'", "").split(", ")))


print(Analysis.load_and_analyse_results("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/linear_regression_feature_search.csv"))


"""


A fitting function to analyse the relationship between the linear regression fit to the data for different features.

The brute force treatment of trying all features suggests for the best fit to the data is a rolling 24 hour std, rolling 24 mean and lag_1. 

The equal best included the use of the "day" feature (feature fitting days). This was done for the Global_active_power property of the data.

Maybe including the other data in the rolling averages, such as relating voltage to global_active_power? or reactive power and global power? 




"""