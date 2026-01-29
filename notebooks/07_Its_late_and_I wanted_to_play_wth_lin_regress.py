
"""

Lets do some simple regression on the data for power predcition 

-------------------------------------

Raw Data

Cleaning / Formatting

Feature Engineering

Train / Validation Split

Baseline Model

Evaluation

----

Linear regression attmpet!

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

Data_frame = loader.load_energy_data("C:/Users/gregr/Current_coding_projects/Git_repo_testing/Training-energy-demand-forecasting-ml/data/raw/household_power_consumption.txt", ';')
df, issues = loader.Data_formatting_clean_reassemble(Data_frame)
print(df.info())


# Assume df_features has 'Datetime' and 'Global_active_power'
df = df.copy()

# features that are then enurmerated so that the date/days/months can be enumerated
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month
df["day"] = df["Datetime"].dt.day
df["year"] = df["Datetime"].dt.year
df["quarter"] = df["Datetime"].dt.quarter



valid_mask = df["Global_active_power"].notna()
df_valid = df[valid_mask].copy()  # This says where there are values that are not valid for cropping out

print('check', type(df_valid), type(df))

X = df_valid[["hour","dayofweek","month",  "day", "year", "quarter", "Datetime"]].copy()  # keep Datetime for plotting
y = df_valid["Global_active_power"] # This is the fitting value in y

print()
print(type(X))
print()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # Splitting the data

# Save timestamps for plotting later
t_train = X_train["Datetime"]
t_test = X_test["Datetime"]

# Drop Datetime for model input as the time data would be super bad (can't fit to datatime64 format)
X_train_model = X_train[["hour","dayofweek","month","day", "year", "quarter"]]
X_test_model  = X_test[["hour","dayofweek","month", "day", "year", "quarter"]]


#   Linear regression !!!!
model = LinearRegression()
model.fit(X_train_model, y_train)

# Predictions
y_train_pred = model.predict(X_train_model)
y_test_pred  = model.predict(X_test_model)

# Metrics --- Currently no analysis or discussion of the data
metrics = {
    "Train MAE": mean_absolute_error(y_train, y_train_pred),
    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "Test MAE": mean_absolute_error(y_test, y_test_pred),
    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
}

print(metrics)


#-----------------PLOTTING"!!--------------------


train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})

plt.figure(figsize=(14,6))
plt.plot(pd.concat([train_df, test_df]).sort_values("Datetime")["Datetime"],
         pd.concat([train_df, test_df]).sort_values("Datetime")["Actual"],
         label="Actual", color="black", alpha=0.6)
plt.scatter(train_df["Datetime"], train_df["Predicted"], label="Train Prediction", color="blue", s=10)
plt.scatter(test_df["Datetime"], test_df["Predicted"], label="Test Prediction", color="red", s=10)
plt.xlabel("Datetime")
plt.ylabel("Global Active Power")
plt.title("Actual vs Predicted Energy Consumption (Hour Feature)")
plt.legend()
plt.tight_layout()
plt.show()



# Define start and end timestamps so that we can get a zoomed in look
start_time = pd.Timestamp("2007-01-01 00:00")
end_time   = pd.Timestamp("2007-03-01 00:00")

# Filter train and test DataFrames zooming
train_zoom = train_df[(train_df["Datetime"] >= start_time) & (train_df["Datetime"] <= end_time)]
test_zoom  = test_df[(test_df["Datetime"] >= start_time) & (test_df["Datetime"] <= end_time)]

fig, ax = plt.subplots(figsize=(12,5))

# Make pretty

locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate()

# Actual
ax.plot(pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Datetime"],
         pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Actual"],
         label="Actual", color="black", alpha=0.6)

# Predictions
ax.scatter(train_zoom["Datetime"], train_zoom["Predicted"], label="Train Prediction", color="blue", s=10)
ax.scatter(test_zoom["Datetime"], test_zoom["Predicted"], label="Test Prediction", color="red", s=10)

ax.set_xlabel("Datetime")
ax.set_ylabel("Global Active Power")
ax.set_title("Zoomed Energy Consumption")
plt.legend()
plt.tight_layout()
plt.show()


# Define start and end timestamps so that we can get a zoomed in look
start_time = pd.Timestamp("2007-03-01 00:00")
end_time   = pd.Timestamp("2007-05-01 00:00")

# Filter train and test DataFrames zooming
train_zoom = train_df[(train_df["Datetime"] >= start_time) & (train_df["Datetime"] <= end_time)]
test_zoom  = test_df[(test_df["Datetime"] >= start_time) & (test_df["Datetime"] <= end_time)]

fig, ax = plt.subplots(figsize=(12,5))

# Make pretty

locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter("%d/%m/%Y %H:%M")
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate()

# Actual
ax.plot(pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Datetime"],
         pd.concat([train_zoom, test_zoom]).sort_values("Datetime")["Actual"],
         label="Actual", color="black", alpha=0.6)

# Predictions
ax.scatter(train_zoom["Datetime"], train_zoom["Predicted"], label="Train Prediction", color="blue", s=10)
ax.scatter(test_zoom["Datetime"], test_zoom["Predicted"], label="Test Prediction", color="red", s=10)

ax.set_xlabel("Datetime")
ax.set_ylabel("Global Active Power")
ax.set_title("Zoomed Energy Consumption")
plt.legend()
plt.tight_layout()
plt.show()


'''
 Not much improvement here by adding lots of additional parameters. 
 
 Taking the extra values from the additional parameters shifted the test data RSME 
Original 3 feature fitting : {'Train MAE': 0.762262308303739, 
                       'Train RMSE': np.float64(1.0110438202084313), 
                       'Test MAE': 0.7625284781065171, 
                       'Test RMSE': np.float64(1.0134337705189904)}

With 6 feature fitting :  {'Train MAE': 0.762262308303739, 
                       'Train RMSE': np.float64(1.0110438202084313), 
                       'Test MAE': 0.7625284781065171, 
                       'Test RMSE': np.float64(1.0134337705189904)}


 
'''