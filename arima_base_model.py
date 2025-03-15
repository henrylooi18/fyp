import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import load_and_preprocess_data 

def run_arima_base_model(years):
    # read dataset directly (bypass scaling since ARIMA doesn't need it)
    read_dataset = pd.read_csv('sustainable_fashion_trends_2024.csv')

    # filter dataset (2010-2024)
    read_dataset_lag = read_dataset[(read_dataset['Year'] >= 2010) & (read_dataset['Year'] <= 2024)]

    # mean carbon footprint per year
    mean_co2_per_year = read_dataset_lag.groupby('Year')['Carbon_Footprint_MT'].mean().reset_index()

    # set year as index
    mean_co2_per_year.set_index('Year', inplace=True)

    # convert to time-series format
    series = mean_co2_per_year['Carbon_Footprint_MT']

    # split into train (80%) and test (20%) data
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # store historical data
    history = list(train)
    predictions = []

    print("\nRunning Base ARIMA Model...")

    # Apply ARIMA with base parameters (no hyperparameter tuning)
    p, d, q = 2, 1, 2  # Default parameters as a base model
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()

        # forecast next value
        output = model_fit.forecast()
        yhat = output[0]

        # store prediction
        predictions.append(yhat)

        # get actual observation
        obs = test.iloc[t]
        history.append(obs)

    # evaluate model performance
    corr_coeff = np.corrcoef(test, predictions)[0, 1]  # correlation coefficient value
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)

    print(f"Correlation Coefficient: {corr_coeff:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

    # plot actual vs predicted values
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train, marker='o', color='green', label="Training Data")
    plt.plot(test.index, test, marker='o', color='blue', label="Test Data")
    plt.plot(test.index, predictions, marker='o', linestyle="dashed", label="ARIMA Predicted Data", color='red')

    # customize plot
    plt.xlabel("Year")
    plt.ylabel("Mean Carbon Footprint (MT)")
    plt.xticks(np.arange(min(years), max(years)+1, 1))
    plt.ylim(230, 270)
    plt.title(f"Base ARIMA({p},{d},{q}) Predicted vs Actual Carbon Footprint")
    plt.legend()

    # show plot
    plt.show()
