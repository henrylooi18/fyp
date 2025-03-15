import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import load_and_preprocess_data

def run_svr_base_model():
    # load preprocessed data
    X_train_scaled, X_test_scaled, y_train, y_test, years, scaler = load_and_preprocess_data()

    # initialize SVR model with base parameters from referenced article
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
    svr_model.fit(X_train_scaled, y_train)
    y_pred = svr_model.predict(X_test_scaled)

    # evaluate model
    corr_coeff = np.corrcoef(y_test, y_pred)[0, 1]  # correlation coefficient value
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nBase SVR Model Results:")
    print(f"Correlation Coefficient: {corr_coeff:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")

    # plot the results (use only test years for better clarity)
    plt.figure(figsize=(10, 6))

    plt.plot(years[:len(y_train)], y_train, color='green', marker='o', label='Training Data')
    plt.plot(years[-len(y_test):], y_test, color='blue', marker='o', label='Test Data')
    plt.plot(years[-len(y_pred):], y_pred, color='red', marker='o', linestyle="dashed", label='SVR Predicted Data')

    # display the plot
    plt.xlabel("Year")
    plt.ylabel("Mean Carbon Footprint (MT)")
    plt.xticks(np.arange(min(years), max(years)+1, 1))
    plt.ylim(230, 270)
    plt.legend()
    plt.title("Base SVR Predicted vs Actual Carbon Footprint")
    plt.show()
