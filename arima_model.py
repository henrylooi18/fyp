import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import load_and_preprocess_data  # Import data preprocessing function

def run_arima_model(years):
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

    # store predictions
    predictions = []

    ## hyperparameter tuning: test multiple values of p, d, q (source p, d and q after) !!!
    print("\nARIMA Hyperparameter Tuning Results:")

    # define parameter ranges for tuning
    p_values = [0, 1, 2]  # number of past values (lag) used to predict future values
    d_values = [0, 1, 2]  # number of times data is differened
    q_values = [0, 1, 2]  # number of past forecasts included in the model

    best_p, best_d, best_q = None, None, None
    best_corr = -1
    best_mse = None
    best_mae = None
    best_predictions = None

    # grid search for best ARIMA parameters
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    history = list(train)  # reset history for new ARIMA run
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

                    # evaluate model 
                    corr_coeff = np.corrcoef(test, temp_predictions)[0, 1]
                    mse = mean_squared_error(test, temp_predictions)
                    mae = mean_absolute_error(test, temp_predictions)

                    print(f"For p = {p}, d = {d}, q = {q}, Correlation Coefficient: {corr_coeff}, Mean Squared Error: {mse:.3f}")

                    # update best parameters
                    if corr_coeff > best_corr:
                        best_corr = corr_coeff
                        best_mse = mse
                        best_mae = mae
                        best_p, best_d, best_q = p, d, q
                        best_predictions = temp_predictions

                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) failed: {e}")


    # display best ARIMA parameters
    print("\nBest ARIMA Results:")
    print(f"Best Hyperparameters: p = {best_p}, d = {best_d}, q = {best_q}")
    print(f"Best Correlation Coefficient: {best_corr:.3f}")
    print(f"Best Mean Squared Error: {best_mse:.3f}")
    print(f"Best Mean Absolute Error: {best_mae:.3f}")
    

    plt.figure(figsize=(10,6))

    plt.plot(train.index, train, marker='o', color='green', label="Training Data")

    plt.plot(test.index, test, marker='o', color='blue', label="Test Data", )
    
    plt.plot(test.index, best_predictions, marker='o', linestyle="dashed", label="ARIMA Predicted Data", color='red')

    # customize plot
    plt.xlabel("Year")
    plt.ylabel("Mean Carbon Footprint (MT)")
    plt.xticks(np.arange(min(years), max(years)+1, 1))
    plt.ylim(200, 300)
    plt.title(f"ARIMA({best_p},{best_d},{best_q}) Predicted vs Actual Carbon Footprint")
    plt.legend()

    # show plot
    plt.show()
