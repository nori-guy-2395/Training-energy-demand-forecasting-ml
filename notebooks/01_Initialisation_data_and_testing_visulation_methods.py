# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:07:34 2026

@author: gregr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

# Data loading from raw data

import energy_ml.data.loader as loader


df = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')

# Data loading cleaning and extraction

Data, issues = loader.extract_clean_numpy(df)  # Returns a dictionary of the data cleaned with NaN values

print('Issues in data:', issues)
print('   #################   \n ')
print('\n ')
print('Available data', Data.keys())
print('   #################   \n ')
print('\n ')

Date_time = loader.Date_Time_formatting(df, 'Date', 'Time')

print('Data loaded and formated for date and time')
print('   #################   \n ')
print('\n ')


