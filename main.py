from data_preprocessing import load_and_preprocess_data
import svr_base_model
import svr_tuned_model 
import arima_base_model
import arima_tuned_model  
# import lstm_model 

def main():
    print("Checking and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, years, scaler = load_and_preprocess_data()
    
    print ("Running SVR base model predictions for carbon emission...")
    svr_base_model.run_svr_base_model()
    print("\nRunning SVR model predictions for carbon emission...")
    svr_tuned_model.run_svr_model(X_train_scaled, X_test_scaled, y_train, y_test, years, scaler)
    
    print("\nRunning ARIMA base model predictions for carbon emission...")
    arima_base_model.run_arima_base_model(years)
    print("\nRunning ARIMA model predictions for carbon emission...")
    arima_tuned_model.run_arima_model(years)

    # print("\nRunning LSTM Model...")
    # lstm_model.run_lstm()

if __name__ == "__main__":
    main()