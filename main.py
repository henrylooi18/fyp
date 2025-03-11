from data_preprocessing import load_and_preprocess_data
import svr_model 
import arima_model  
# import lstm_model 

def main():
    print("Checking and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, years, scaler = load_and_preprocess_data()
    
    print("\nRunning SVR model predictions for carbon emission...")
    svr_model.run_svr_model(X_train_scaled, X_test_scaled, y_train, y_test, years, scaler)
    
    print("\nRunning ARIMA model predictions for carbon emission...")
    arima_model.run_arima_model(years)

    # print("\nRunning LSTM Model...")
    # lstm_model.run_lstm()

if __name__ == "__main__":
    main()