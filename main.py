from data_preprocessing import load_and_preprocess_data
import svr_model  # Runs the SVR model
# import arima_model  # Uncomment this when implementing ARIMA
# import lstm_model  # Uncomment this when implementing LSTM

def main():
    print("Checking and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, years, scaler = load_and_preprocess_data()
    
    print("\nRunning SVR Model...")
    svr_model.run_svr(X_train_scaled, X_test_scaled, y_train, y_test, years, scaler)

    # Uncomment to run additional models
    # print("\nRunning ARIMA Model...")
    # arima_model.run_arima()

    # print("\nRunning LSTM Model...")
    # lstm_model.run_lstm()

if __name__ == "__main__":
    main()