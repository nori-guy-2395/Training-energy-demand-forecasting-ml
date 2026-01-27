
import numpy as np
import matplotlib.pyplot as plt
import energy_ml.data.loader as loader
import energy_ml.analysis.plotting as plotting
import energy_ml.analysis.Basic_stats_scripts as Base_stats
import pandas as pd

df = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')

# Data loading cleaning and extraction

Data, issues = loader.extract_clean_numpy(df)  # Returns a dictionary of the data cleaned with NaN values

print('Issues in data:', issues) # show the issues in the data
print('   #################   \n ')
print('\n ')
print('Available data', Data.keys(), type(Data)) # List the headers in the data
print('   #################   \n ')
print('\n ')

Date_time = loader.Date_Time_formatting(df, 'Date', 'Time') # find or combine the Date and Time headers into a date and time 
                                                            # should be in the format "%d/%m/%Y %H:%M:%S"
print('Data loaded and formated for date and time')  #Think that this is superflucious
print('   #################   \n ')
print('\n ')


time_and_date, locator, formatter  = loader.build_datetime_numpy(df, 'Date', 'Time') # Demonstrate the data has been loaded 
print(time_and_date, type(time_and_date))

print('-----------------------')
print('\n ')
print('\n ')

time_and_data_dic = {'Datetime': time_and_date}
# print(time_and_data_dic)
# cleaned_df = pd.DataFrame(time_and_data_dic)

# print(cleaned_df)

#### plotting of the data

# plotting.create_basic_figure_of_data(time_and_date, Data['Global_active_power'], 'Date and time', 'Power',  Date=True) 

# Basic plot of data against dates
# with basic formatting of the data in the 

# Continuing the plotting for all of the headers 

# plotting.create_basic_figure_of_data(time_and_date, Data['Global_active_power'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Global_reactive_power'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Voltage'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Global_intensity'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Sub_metering_1'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Sub_metering_2'], 'Date and time', 'Power',  Date=True) 
# plotting.create_basic_figure_of_data(time_and_date, Data['Sub_metering_3'], 'Date and time', 'Power',  Date=True) 

# Data is in and visible. Analysis of the data can continue. New script for means/mode/medium/min/max ?

# print(Data['Voltage'][1:5]) # Super simple mean, but a better way to deal with this might be using a new pandas dataframe

# print(plotting.mean_of_data(Data['Voltage'], 1, 5)) 

# reonsruct the dataframe, clunky, work it out then write a function

# time_and_data_dic = {'Datetime': time_and_date}
# print(time_and_data_dic)
# cleaned_df = pd.DataFrame(time_and_data_dic)

# print(cleaned_df)

cleaned_df = pd.DataFrame(Data)

print(cleaned_df)

cleaned_df['Datetime'] = time_and_date

print(cleaned_df)

