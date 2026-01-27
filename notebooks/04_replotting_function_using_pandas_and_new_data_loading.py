
'''

Replotting the data from the old method. 

Redoing the analysis and taking into account the pandas framework

The time ranges for different slices are accounted for and analysis of the data in these periods are carried out.


'''


import numpy as np
import matplotlib.pyplot as plt
import energy_ml.data.loader as loader
import energy_ml.analysis.plotting as plotting
import energy_ml.analysis.Basic_stats_scripts as Base_stats
import pandas as pd


df = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
Data_frame, issues = loader.Data_formatting_clean_reassemble(df)

print(Data_frame, type(Data_frame))

# print(Data_frame['Voltage'], type(Data_frame['Voltage']) )

# print(Data_frame['Datetime'], type(Data_frame['Datetime']) )

print(type(Data_frame['Datetime']), type(Data_frame['Voltage']))


print('#########')
print(Data_frame.info())
print('#########')
# print(Data_frame.dtypes) 
# print('#########')


# Basic plotting scripts for the plotting of all of the data

plotting.create_basic_figure_of_data(Data_frame['Datetime'],  Data_frame['Voltage'], 'Date and time [days/months/years hours/seconds]', 'Voltage', Date=True)

plotting.create_basic_figure_for_each_column_against_datetime(Data_frame)

##  Now I want to add basic statistics (mean, max, mode, median, min, deviation for all of the data)

# Data_frame.set_index('Datetime', inplace=True)

# ## Daily mean (already daily here, but useful for longer data)
# weekly_mean = Data_frame['Voltage'].resample('W').mean()
# print('Weekly mean: ',weekly_mean)

# ## Weekly statistics
# Yearly_stats = Data_frame['Voltage'].resample('YE').agg(['mean', 'median', 'min', 'max', 'std'])
# print(Yearly_stats)



# ## Okay I think this works for the defined case. Days are given in D, W, and Years. 

# ## This can maybe implemented into the basic stats script as a function

# Base_stats.Information_data_analysis(Data_frame, 'Voltage', 'YE')

# That felt clunky, so found a new way. The datetime64 format is useful. Check the time is indexed in the function and then implement. 

# Set start date and end dates with the values returned. Seems to work well. 

print(Base_stats.time_period_stats(Data_frame, 'Datetime', 'Voltage', '2007-01-01', '2011-01-01'))

# This seems fine. All objecives seem to be met 

