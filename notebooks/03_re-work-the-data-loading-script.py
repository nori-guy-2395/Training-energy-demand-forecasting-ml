"""
The loaing script fel clunky. 
I want to make download the data. 
extract to dictionary the values for numpy treatment. 
Extract date and time values 
Combine the date and time values to a datetime dictionary with numpy valid dates
Recombine all the data such that I have a new pandas dataframe. 

This should make analysis a lot easier. 

The script should call/check for errors and fill NaN values where numeric data is not available

"""

import numpy as np

import pandas as pd
from datetime import datetime, timedelta

# Data loading from raw data

import energy_ml.data.loader as loader # The same loader function is useable to get the raw data

df = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')

#This has the separable ; already included and produces a good data frame
####

# Now I need to go through each columnn with numeric data and clean them
# First find columns with numeric numpy relevant data

# data = df.replace('?', np.nan)

# output = {}
# issues = {}

# for col in data.columns:
#     series = data[col]

#     # # ---- DATETIME ----
#     # if col.lower().startswith("datetime"):
#     #     parsed = pd.to_datetime(series, errors="coerce")

#     # ---- DATE ----
#     if col.lower().startswith("Date"):
#         parsed = pd.to_datetime(series, errors="coerce", dayfirst=True).dt.date

#     # ---- TIME ----
#     elif col.lower().startswith("Time"):
#         parsed = pd.to_datetime(series, errors="coerce").dt.time

#     # ---- NUMERIC ----
#     else:
#         parsed = pd.to_numeric(series, errors="coerce")

#     # ---- Track issues ----
#     bad_mask = parsed.isna() & series.notna()
#     if bad_mask.any():
#         issues[col] = series[bad_mask]

#     # ---- Convert to numpy ----
#     output[col] = parsed.to_numpy()



# # This seems to work, so now I need to extract, check and convert the time and date data into a datetime column and add that to the df

# data_datetime = pd.to_datetime(data['Date'] + " " + data['Time'])


# data['Datetime'] = data_datetime
# print('Data after time addition', data, '\n \n' 'issues', issues, type(data))

# print('Data after time addition headers', data.keys())

# #  I want this column to move to the first column


# col_last = data.pop('Datetime')


# data.insert(0, 'Datetime', col_last)


# print(type(data))
# Now I'm implementing this in the loader.py script so that I get a clean imported pandas frame


# New function assembling the above into a clean dataframe inside the loadp.py

import energy_ml.data.loader as loader

Data_frame = loader.Data_formatting_clean_reassemble(df)

print(Data_frame)
