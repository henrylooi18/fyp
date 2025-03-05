import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # read dataset
    read_dataset = pd.read_csv('sustainable_fashion_trends_2024.csv')

    # clean selected column
    column_name = 'Carbon_Footprint_MT'
    
    # check missing values
    missing_values = read_dataset[column_name].isnull().sum()
    print("Missing Values from Dataset:", missing_values)

    ## lag features (filter dataset to relevant years)
    read_dataset_lag = read_dataset[(read_dataset['Year'] >= 2010) & (read_dataset['Year'] <= 2024)]

    # mean carbon footprint per year
    mean_co2_per_year = read_dataset_lag.groupby('Year')['Carbon_Footprint_MT'].mean().reset_index()

    # convert year to index
    mean_co2_per_year['Year_Index'] = mean_co2_per_year['Year'] - mean_co2_per_year['Year'].min()

    # store data in X and y
    X = mean_co2_per_year[['Year_Index']].values
    y = mean_co2_per_year['Carbon_Footprint_MT'].values

    # split to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

    # normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, mean_co2_per_year['Year'].values, scaler