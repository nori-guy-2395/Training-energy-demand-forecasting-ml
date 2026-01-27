import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

def load_energy_data(path: str, sepa: str) -> pd.DataFrame:
    """
    Load and return raw energy consumption data.
    """
    df = pd.read_csv(path, sep=sepa)
    
    print(" Top 3 rows of data input", df.head(3)) # show top rows

    return df


def Date_Time_formatting(data, column_date: str, column_time: str, Format_date= '%d/%m/%Y', Format_time= '%H:%M:%S'):
    """
    

        Parameters
        ----------
        data : Pandas dataframe
            Dataframe from pandas import.
        column_date : str
            Title of the column data for the dates
        column_time : str
            Title of the column data for the Times
        Format_date : TYPE, optional
            Format of the dates. The default is '%d/%m/%Y'.
        Format_time : TYPE, optional
            Format of the Times. The default is '%H:%M:%S'.
    
        Returns
        -------
        time_and_data: numpy array
            combined date and time numpy array.
        locator : ?
            matplotlib locator option already calculated (for cleaner plotting)
        formatter : ?
            matplotlib formatting option already calculated (for cleaner plotting)

    """


    data = pd.to_datetime(data[column_date] + " " + data[column_time])

    
    print('checking format load:', data[0])
    time_and_data = pd.to_datetime(data, format='%d/%m/%Y %H:%M:%S').to_numpy()
    
    # Use AutoDateLocator and AutoDateFormatter
    locator = mdates.AutoDateLocator()      # automatically choose tick positions
    formatter = mdates.AutoDateFormatter(locator)
    # print(type(time_and_data))
    return time_and_data, locator, formatter


def extract_clean_numpy(data, null_values=(" ", "", "NA", "?")):
    data = data.replace(null_values, np.nan)

    output = {}
    issues = {}

    for col in data.columns:
        series = data[col]

        # ---- DATETIME ----
        if col.lower().startswith("datetime"):
            parsed = pd.to_datetime(series, errors="coerce")

        # ---- DATE ----
        elif col.lower().startswith("Date"):
            parsed = pd.to_datetime(series, errors="coerce").dt.date

        # ---- TIME ----
        elif col.lower().startswith("Time"):
            parsed = pd.to_datetime(series, errors="coerce").dt.time

        # ---- NUMERIC ----
        else:
            parsed = pd.to_numeric(series, errors="coerce")

        # ---- Track issues ----
        bad_mask = parsed.isna() & series.notna()
        if bad_mask.any():
            issues[col] = series[bad_mask]

        # ---- Convert to numpy ----
        output[col] = parsed.to_numpy()

    return output, issues


def build_datetime_numpy(data, column_date, column_time):
    """
    Combine date and time columns into a numpy datetime64 array
    and return matplotlib date locator/formatter.
    """

    # Combine as strings (safe with NaN)
    combined = (
        data[column_date].astype(str).str.strip() + " " +
        data[column_time].astype(str).str.strip()
    )

    # Convert to datetime, coercing failures to NaT
    datetime_series = pd.to_datetime(
        combined,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # Debug check (first valid value)
    print("checking format load:", datetime_series.dropna().iloc[0])

    # Convert to numpy datetime64[ns]
    time_and_date = datetime_series.to_numpy()

    # Matplotlib helpers
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)

    return time_and_date, locator, formatter 



def Data_formatting_clean_reassemble2(data_frame):
    '''
    

    Parameters
    ----------
    data_frame : dict
        Initial imported data frame which has had the separators accounted for.
        

    Returns
    -------
    pandas dataframe dict. with clean numpy values in value columns, clean dates in date and time columns, 
    additional combined date time column at 0th index

    '''
    null_values=(" ", "", "NA", "?")
    
    data_frame = data_frame.replace(null_values, np.nan)

    output = {}
    issues = {}

    for col in data_frame.columns:
        series = data_frame[col]

        # # ---- DATETIME ----
        # if col.lower().startswith("Datetime"):
        #     parsed = pd.to_datetime(series, errors="coerce")

        # ---- DATE ----
        if col.lower().startswith("Date"):
            parsed = pd.to_datetime(series, errors="coerce").dt.date

        # ---- TIME ----
        elif col.lower().startswith("Time"):
            parsed = pd.to_datetime(series, errors="coerce").dt.time

        # ---- NUMERIC ----
        else:
            parsed = pd.to_numeric(series, errors="coerce")

        # ---- Track issues ----
        bad_mask = parsed.isna() & series.notna()
        if bad_mask.any():
            issues[col] = series[bad_mask]

        # ---- Convert to numpy ----
        output[col] = parsed.to_numpy()

    data_frame_clean = pd.DataFrame(output)

    # This seems to work, so now I need to extract, check and convert the time and date data into a datetime column and add that to the df

    data_datetime = pd.to_datetime(data_frame_clean['Date'] + " " + data_frame_clean['Time'], dayfirst=True)


    data_frame_clean['Datetime'] = data_datetime
    # print('Data after time addition', data_frame_clean, '\n \n' 'issues', issues, type(data_frame_clean))

    # print('Data after time addition headers', data_frame_clean.keys())

    #  I want this column to move to the first column


    col_last = data_frame_clean.pop('Datetime')


    data_frame_clean.insert(0, 'Datetime', col_last)


    # print(type(data_frame_clean))
    
    
    
    return data_frame_clean



def Data_formatting_clean_reassemble(df):
    """
    Clean dataframe:
    - Convert date/time/numeric columns
    - Combine Date + Time into Datetime
    - Move Datetime to first column
    - Track parsing issues
    """

    null_values = (" ", "", "NA", "?")
    df = df.replace(null_values, np.nan)

    clean_df = pd.DataFrame(index=df.index)
    issues = {}

    for col in df.columns:
        series = df[col]
        col_lc = col.lower()

        # ---- DATE ----
        if col_lc.startswith("date"):
            parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce", dayfirst=True) # Format specified is hardcoded !!! check for alternative forms

        # ---- TIME ----
        elif col_lc.startswith("time"):
            parsed = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")  # Format specified is hardcoded !!! check for alternative forms

        # ---- NUMERIC ----
        else:
            parsed = pd.to_numeric(series, errors="coerce")

        # ---- Track issues ----
        bad_mask = parsed.isna() & series.notna()
        if bad_mask.any():
            issues[col] = series[bad_mask]

        clean_df[col] = parsed

    # ---- Combine Date + Time ----
    if "Date" in clean_df.columns and "Time" in clean_df.columns:
        clean_df["Datetime"] = pd.to_datetime(
            clean_df["Date"].dt.strftime("%Y-%m-%d") + " " +
            clean_df["Time"].dt.strftime("%H:%M:%S"),
            errors="coerce"
        )

        # ---- Move Datetime to front ----
        col = clean_df.pop("Datetime")
        clean_df.insert(0, "Datetime", col)

    return clean_df, issues