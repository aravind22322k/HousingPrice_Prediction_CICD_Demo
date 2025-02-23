# feature_engineering.py
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("data\housing_data.csv")

# Feature Engineering

# 1. Price per square foot: helps understand pricing relative to size
data['price_per_sqft'] = data.apply(lambda row: row['price'] / row['sqft'] if row['sqft'] > 0 else 0, axis=1)

# 2. Bedrooms per square foot: provides insight into space allocation per bedroom
data['bedrooms_per_sqft'] = data.apply(lambda row: row['bedrooms'] / row['sqft'] if row['sqft'] > 0 else 0, axis=1)

# 3. Log transformation of sqft and price to reduce skewness (useful if the distributions are highly skewed)
data['log_sqft'] = np.log(data['sqft'])
data['log_price'] = np.log(data['price'])

# 4. Interaction term: product of sqft and bedrooms (captures potential interaction between size and number of bedrooms)
data['sqft_bedrooms'] = data['sqft'] * data['bedrooms']

# Save the engineered dataset to a new CSV file
output_path = "data\housing_engineered.csv"
data.to_csv(output_path, index=False)

print(f"Feature engineering complete. Engineered dataset saved to {output_path}")
