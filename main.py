import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

# read dataset
read_dataset = pd.read_csv('sustainable_fashion_trends_2024.csv')

# clean selected columns
column_name = 'Carbon_Footprint_MT'

# check rows with missing values
missing_values = read_dataset[column_name].isnull().sum()
print ("Missing Values from Dataset: ", missing_values)

## lag features
# target results of 2014 (2010 to 2013)
read_dataset_lag = read_dataset[(read_dataset['Year'] >= 2010) & (read_dataset['Year'] <= 2014)]

# mean carbon footprint per year
mean_co2_per_year = read_dataset_lag.groupby('Year')['Carbon_Footprint_MT'].mean().reset_index()

# convert year into index
mean_co2_per_year['Year_Index'] = mean_co2_per_year['Year'] - mean_co2_per_year['Year'].min()

plt.figure(figsize=(8,5))
plt.plot(mean_co2_per_year['Year'], mean_co2_per_year['Carbon_Footprint_MT'], marker='o', linestyle='-', color='b')

# plotting target results of 2014 (2010 to 2013)
# plt.xlabel("Year")
# plt.xticks(mean_co2_per_year['Year'].astype(int)) 
# plt.ylabel("Mean Carbon Footprint (MT)")
# plt.title("Target Mean Carbon Footprint Per Year (2010-2014)")
# plt.grid(True)
# plt.show()

# SVR Model preparation
# X = mean_co2_per_year[['Year_Index']].values  # numerical year index (required by SVR)
# y = mean_co2_per_year['Carbon_Footprint_MT'].values  # target values

# print("\nPrepared Data for SVR Model:")
# print(mean_co2_per_year)

# Set Year as index for ARIMA
mean_co2_per_year.set_index('Year', inplace=True)

# Convert to time-series format
series = mean_co2_per_year['Carbon_Footprint_MT']

# Split data into 80% training and 20% testing
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Store history of known values
history = list(train)

# Store predictions
predictions = []

# **Apply ARIMA model iteratively (Rolling Forecast)**
print("\nARIMA Predictions vs. Actual Values:")
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))  # (p=5, d=1, q=0) from ARIMA(p,d,q)
    model_fit = model.fit()
    
    # Forecast next step
    output = model_fit.forecast()
    yhat = output[0]
    
    # Store prediction
    predictions.append(yhat)
    
    # Get actual observation
    obs = test.iloc[t]
    history.append(obs)

    # Print predicted vs actual
    print(f"Predicted={yhat:.2f}, Expected={obs:.2f}")

# **Evaluate Model Performance**
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
print("\nARIMA Model Evaluation:")
print(f"Mean Absolute Error (MAE): ", mae)
print(f"Mean Squared Error (MSE): ", mse)

# **Plot Actual vs. Predicted Values**
plt.figure(figsize=(8,5))
plt.plot(test.index, test, marker='o', label="Actual", color='blue')
plt.plot(test.index, predictions, marker='s', linestyle="dashed", label="Predicted", color='red')

# Customize plot
plt.xlabel("Year")
plt.ylabel("Mean Carbon Footprint (MT)")
plt.title("ARIMA Predicted vs Actual Carbon Footprint (2014)")
plt.legend()
plt.grid(True)

# Show plot
plt.show()