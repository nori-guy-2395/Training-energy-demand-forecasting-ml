''' 

We have the model from the linear regression. Load, setup, analyse, and plot. 

We can swap the setup from linear to the random forest regression case.

The random forest model can be better than linear regression at encountering errors and more robust against overfitting.

The model should be able to return general interpretations as part of its return

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


model = LinearRegression()

#   SWAPPED OUT THE MDOEL FOR A RANDOM FOREST REGRESSOR  - "STANDARD SETTINGS" !
# model = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=None,
#     random_state=42,
#     n_jobs=-1
# )

"""

Linear regression 

{'Train MAE': 0.762262308303739, 
 'Train RMSE': np.float64(1.0110438202084313), 
 'Test MAE': 0.7625284781065171, 
 'Test RMSE': np.float64(1.0134337705189904)}


Random forest regressor

{'Train MAE': 0.5961070530571221, 
 'Train RMSE': np.float64(0.8692514979178609), 
 'Test MAE': 0.5971279673091375, 
 'Test RMSE': np.float64(0.8727702151423351)}


BAsic model improves the MAE and RMSE over the linear regression model, 

but notable for slightly longer run times (not traked accurately, but can tell)


"""

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
plt.tight_layout()+
plt.show()


