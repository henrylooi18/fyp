import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_and_preprocess_data  # Import data preprocessing function

def run_arima():
    # load dataset (arima only needs years variable to predict)
    _, _, _, _, years, _ = load_and_preprocess_data()

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

    # split into train (80%) and test (20%) sets
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    # store historical data
    history = list(train)

    # store predictions
    predictions = []

    print("\nARIMA Hyperparameter Tuning Results:")

    # define parameter ranges for tuning
    p_values = [0, 1, 2, 3, 4, 5]  # Auto-regression !!!
    d_values = [0, 1, 2]           # Differencing !!!
    q_values = [0, 1, 2]           # Moving average !!!

    best_p, best_d, best_q = None, None, None
    best_mse = float("inf")
    best_predictions = None

    # grid search for best ARIMA parameters
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    history = list(train)  # Reset history for each ARIMA run !!!
                    temp_predictions = []

                    for t in range(len(test)):
                        model = ARIMA(history, order=(p, d, q))
                        model_fit = model.fit()

                        # forecast next value
                        output = model_fit.forecast()
                        yhat = output[0]

                        # store prediction
                        temp_predictions.append(yhat)

                        # get actual observation
                        obs = test.iloc[t]
                        history.append(obs)

                    # evaluate model performance
                    mse = mean_squared_error(test, temp_predictions)
                    print(f"ARIMA({p},{d},{q}) - MSE: {mse:.3f}")

                    # update best parameters
                    if mse < best_mse:
                        best_mse = mse
                        best_p, best_d, best_q = p, d, q
                        best_predictions = temp_predictions

                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) failed: {e}")

    print("\nBest ARIMA Hyperparameters Found:")
    print(f"Best p = {best_p}, Best d = {best_d}, Best q = {best_q}")
    print(f"Best Mean Squared Error = {best_mse:.3f}")

    # plot actual vs best predicted values
    plt.figure(figsize=(10,6))
    plt.plot(test.index, test, marker='o', label="Actual", color='blue')
    plt.plot(test.index, best_predictions, marker='s', linestyle="dashed", label="Best ARIMA Predicted", color='red')

    # customize plot
    plt.xlabel("Year")
    plt.ylabel("Mean Carbon Footprint (MT)")
    plt.title(f"ARIMA({best_p},{best_d},{best_q}) Predicted vs Actual Carbon Footprint")
    plt.legend()
    plt.grid(True)

    # show plot
    plt.show()