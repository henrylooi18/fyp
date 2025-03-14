import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocessing import load_and_preprocess_data

def run_svr_model(X_train_scaled, X_test_scaled, y_train, y_test, years, scaler):
    # load preprocessed data
    X_train_scaled, X_test_scaled, y_train, y_test, years, scaler = load_and_preprocess_data()

    ## hyperparameter tuning: test multiple values of C and gamma (given in the article)
    gamma_list = ['auto', 0.5, 0.3, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001]
    c_list = [1, 10, 1e2, 1e3, 1e4, 1e5]

    best_c = None
    best_gamma = None
    best_corr = -1  # start with lowest correlation (perfect negative correlation) as the base to print the corr_coeff value
    best_mse = None
    best_mae = None
    best_y_pred = None

    print("\nSVR Hyperparameter Tuning Results:")

    # nested for loop - grid search for best SVR C and gamma parameters combination
    for c in c_list:  # regularization parameter (trade-off between achieving low error and maintaining a simple model)
        for g in gamma_list:  # kernel coefficient (how much influence a single training example has)
            svr_model = SVR(kernel='rbf', C=c, gamma=g)
            svr_model.fit(X_train_scaled, y_train)
            y_pred = svr_model.predict(X_test_scaled)

            # evaluate model
            corr_coeff = np.corrcoef(y_test, y_pred)[0, 1]  # correlation coefficient value
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            print(f"For C = {c}, gamma = {g}, Correlation Coefficient = {corr_coeff:.3f}, Mean Squared Error = {mse:.3f}")

            # looking for the best model
            if corr_coeff > best_corr:
                best_corr = corr_coeff
                best_c = c
                best_gamma = g
                best_mse = mse
                best_mae = mae
                best_y_pred = y_pred

    # display best SVR parameters
    print("\nBest SVR Results:")
    print(f"Best Hyperparameters: C = {best_c}, gamma = {best_gamma}")
    print(f"Best Correlation Coefficient: {best_corr:.3f}")
    print(f"Best Mean Squared Error: {best_mse:.3f}")
    print(f"Best Mean Absolute Error: {best_mae:.3f}") 

    ## plot the results (use only test years for better clarity)
    plt.figure(figsize=(10, 6))
    plt.plot(years[:len(y_train)], y_train, color='green', marker='o', label='Training Data')
    plt.plot(years[-len(y_test):], y_test, color='blue', marker='o', label='Test Data')
    plt.plot(years[-len(best_y_pred):], best_y_pred, color='red', marker='o', linestyle="dashed", label='SVR Predicted Data')

    # display the plot
    plt.xlabel("Year")
    plt.ylabel("Mean Carbon Footprint (MT)")
    plt.xticks(np.arange(min(years), max(years)+1, 1))
    plt.ylim(230, 270)
    plt.legend()
    plt.title("SVR Predicted vs Actual Carbon Footprint (Test Data)")
    plt.show()