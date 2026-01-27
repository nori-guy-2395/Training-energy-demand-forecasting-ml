
"""
Plotting time intervals properly and neatly.

"""

import numpy as np
import matplotlib.pyplot as plt
import energy_ml.data.loader as loader
import energy_ml.analysis.plotting as plotting
import energy_ml.analysis.Basic_stats_scripts as Base_stats
import pandas as pd

# generic import of the data

df = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
Data_frame, issues = loader.Data_formatting_clean_reassemble(df)

plotting.create_basic_figure_of_data(Data_frame['Datetime'],  Data_frame['Voltage'], 'Date and time [days/months/years hours/seconds]', 'Voltage', Date=True)


plotting.figure_plotting_time_interval(Data_frame, 'Datetime', 'Voltage', 'Date and time [days/months/years hours/seconds]', 'Voltage', Date = True, start = '01-01-2007', end = '02-01-2007')


