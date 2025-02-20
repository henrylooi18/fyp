import pandas as pd

# read dataset
read_dataset = pd.read_csv('sustainable_fashion_trends_2024.csv')

# clean selected columns
column_name = 'Carbon_Footprint_MT'

# check rows with missing values
missing_values = read_dataset[column_name].isnull().sum()
print ("Missing Values in '{column_name}': ", missing_values)

# selected metrics for prediction
selected_columns = ['Carbon_Footprint_MT']

# descriptive statistics
descriptive_statistics = read_dataset[selected_columns].describe()

# calculating mean, variance, standard deviation, minimum & maximum values of selected column
mean = read_dataset[selected_columns].mean()
variance = read_dataset[selected_columns].var()
standard_deviation = read_dataset[selected_columns].std()
minimum = read_dataset[selected_columns].min()
maximum = read_dataset[selected_columns].max()

# print raw descriptive statistics
print("Descriptive Statistics: \n", descriptive_statistics)
print("\nMean: ", mean)
print("\nVariance: ", variance)
print("\nStandard Deviation: ", standard_deviation)
print("\nMinimum: ", minimum)
print("\nMaximum: ", maximum)
