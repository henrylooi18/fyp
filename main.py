import pandas as pd
import matplotlib.pyplot as plt

# read dataset
read_dataset = pd.read_csv('sustainable_fashion_trends_2024.csv')

# clean selected columns
column_name = 'Carbon_Footprint_MT'

# check rows with missing values
missing_values = read_dataset[column_name].isnull().sum()
print ("Missing Values from Dataset: ", missing_values)

## lag features
# target results of 2014 (2010 to 2013)
read_dataset_lag = read_dataset[(read_dataset['Year'] >= 2010) & (read_dataset['Year'] <= 2014)]

# mean carbon footprint per year
mean_co2_per_year = read_dataset_lag.groupby('Year')['Carbon_Footprint_MT'].mean().reset_index()

# convert year into index
mean_co2_per_year['Year_Index'] = mean_co2_per_year['Year'] - mean_co2_per_year['Year'].min()

plt.figure(figsize=(8,5))
plt.plot(mean_co2_per_year['Year'], mean_co2_per_year['Carbon_Footprint_MT'], marker='o', linestyle='-', color='b')

# plotting target results of 2014 (2010 to 2013)
plt.xlabel("Year")
plt.xticks(mean_co2_per_year['Year'].astype(int)) 
plt.ylabel("Mean Carbon Footprint (MT)")
plt.title("Target Mean Carbon Footprint Per Year (2010-2014)")
plt.grid(True)
plt.show()

# SVR Model preparation
X = mean_co2_per_year[['Year_Index']].values  # numerical year index (required by SVR)
y = mean_co2_per_year['Carbon_Footprint_MT'].values  # target values

print("\nPrepared Data for SVR Model:")
print(mean_co2_per_year)