# -*- coding: utf-8 -*-
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


valid_mask = df["Global_active_power"].notna()
df_valid = df[valid_mask].copy()  # This says where there are values that are not valid for cropping out


X = df_valid[["hour","dayofweek","month", "Datetime"]].copy()  # keep Datetime for plotting
y = df_valid["Global_active_power"] # This is the fitting value in y


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # Splitting the data

# Save timestamps for plotting later
t_train = X_train["Datetime"]
t_test = X_test["Datetime"]

# Drop Datetime for model input as the time data would be super bad (can't fit to datatime64 format)
X_train_model = X_train[["hour","month","dayofweek"]]
X_test_model  = X_test[["hour","month","dayofweek"]]


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
end_time   = pd.Timestamp("2007-01-07 00:00")

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














# target_col="Global_active_power"
# datetime_col = 'Datetime'

# df = Data_frame.copy()
# df["hour"] = df[datetime_col].dt.hour
# X = df[["hour", "dayofweek", "month"]].copy()
# X["Datetime"] = df["Datetime"]  # keep it for plotting


# X = Data_frame[["hour"]]
# y = Data_frame[target_col]

# X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )


# t_train = X_train["Datetime"]
# t_test = X_test["Datetime"]

# # Drop Datetime from features before fitting
# X_train_model = X_train.drop(columns=["Datetime"])
# X_test_model = X_test.drop(columns=["Datetime"])

# model = LinearRegression()

# model.fit(X_train, y_train)
# predictions = model.predict(X_test)


# metrics = {
#         "MAE": mean_absolute_error(y_test, predictions),
#         "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
#     }

# train_plot_df = pd.DataFrame({
#     "Datetime": t_train,
#     "Actual": y_train,
#     "Predicted": y_train_pred,
#     "Set": "Train"
# })


# # Feature extraction
# df["hour"] = df["Datetime"].dt.hour
# df["dayofweek"] = df["Datetime"].dt.dayofweek
# df["month"] = df["Datetime"].dt.month

# print('Adjusted data', df.info())


# # Keep Datetime for plotting
# X = df[["hour", "dayofweek", "month", "Datetime"]]
# y = df["Global_active_power"]

# print('X', X.info())
# print('y', y.info())

# # Boolean mask for non-NaN targets
# valid_mask = y.notna()

# print('valid mask', valid_mask)

# # Keep only rows where y is not NaN
# X_valid = X[valid_mask].copy()
# y_valid = y[valid_mask].copy()

# # If Datetime is in X, also split it out
# t_valid = X_valid["Datetime"]
# X_valid_model = X_valid.drop(columns=["Datetime"])  # numeric features only


# print('T valid', t_valid.info())
# print('X_valid model data', X_valid_model)
# print('X valid model', X_valid_model.info())

# # Train/test split
# X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
#     X_valid_model, y_valid, t_valid, test_size=0.2, random_state=42
# )

# print('T train', t_train.info())
# print('X train after model split', X_train.info())

# # Extract timestamps for plotting
# t_train = X_train["Datetime"]

# print('T train', t_train.info())

# t_test = X_test["Datetime"]

# # Model features only
# X_train_model = X_train.drop(columns=["Datetime"])
# X_test_model  = X_test.drop(columns=["Datetime"])

# # print(t_train)
# # print(X_train_model)

# # Initialize the model
# model = LinearRegression()

# # Train on numeric features only
# model.fit(X_train_model, y_train)

# # Predict on both train and test sets
# y_train_pred = model.predict(X_train_model)
# y_test_pred  = model.predict(X_test_model)


# metrics = {
#     "Train MAE": mean_absolute_error(y_train, y_train_pred),
#     "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
#     "Test MAE": mean_absolute_error(y_test, y_test_pred),
#     "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
# }
# print(model.coefs_)
# print(metrics)

# train_df = pd.DataFrame({"Datetime": t_train, "Actual": y_train, "Predicted": y_train_pred})
# test_df  = pd.DataFrame({"Datetime": t_test,  "Actual": y_test,  "Predicted": y_test_pred})

# plt.figure(figsize=(14,6))
# plt.plot(pd.concat([train_df, test_df]).sort_values("Datetime")["Datetime"],
#          pd.concat([train_df, test_df]).sort_values("Datetime")["Actual"], label="Actual", color="black", alpha=0.6)
# plt.scatter(train_df["Datetime"], train_df["Predicted"], label="Train Prediction", color="blue", s=10)
# plt.scatter(test_df["Datetime"], test_df["Predicted"], label="Test Prediction", color="red", s=10)
# plt.xlabel("Datetime")
# plt.ylabel("Global Active Power")
# plt.title("Actual vs Predicted Energy Consumption")
# plt.legend()
# plt.tight_layout()
# plt.show()





# print('features', df_features)
# model, metrics = train_baseline_model(df_features, target_col="Global_active_power")  # implementing / trialling methods to get the data to work 

# # NEED to look again at this assumption of removing NaN data. 
# print(model.coef_)
# print(metrics)

# X = df_model[["hour", "dayofweek", "month"]]
# y = df_model[target_col]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# metrics = {
#         "MAE": mean_absolute_error(y_test, predictions),
#         "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
#     }





# def add_time_features(df, datetime_col="Datetime"):
#     df = df.copy()

#     df["hour"] = df[datetime_col].dt.hour
#     df["dayofweek"] = df[datetime_col].dt.dayofweek
#     df["month"] = df[datetime_col].dt.month

#     return df

# def train_baseline_model(df, target_col, Nan_remove= False, Imputer=False, pipeline = False):
    
#     if Nan_remove == True and Imputer == False:
#         df_model = df.dropna(subset=[target_col, "hour", "dayofweek", "month"])
#     elif Nan_remove == False and Imputer == True:
#         imputer = SimpleImputer(strategy="mean")
#         df_model = imputer.fit_transform(df[target_col])
#     else:
#         df_model = df
    
#     X = df_model[["hour", "dayofweek", "month"]]
#     y = df_model[target_col]
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     model = LinearRegression()
#     if pipeline ==True:
#         pipeline = Pipeline([
#                             ("imputer", SimpleImputer(strategy="mean")),
#                             ("model", LinearRegression())
#                         ])
#         pipeline.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         model = pipeline
#     else:
#         model.fit(X_train, y_train)

#         predictions = model.predict(X_test)

#     metrics = {
#         "MAE": mean_absolute_error(y_test, predictions),
#         "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
#     }

#     return model, metrics


