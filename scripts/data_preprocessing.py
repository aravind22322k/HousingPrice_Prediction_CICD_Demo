# data preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("C:\\dat1\\housing.csv")
print("Initial data:")
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# Fill missing values (if any) using median strategy
if missing_values.any():
    imputer = SimpleImputer(strategy="median")
    data[['sqft', 'bedrooms', 'price']] = imputer.fit_transform(data[['sqft', 'bedrooms', 'price']])
    print("\nMissing values filled with median.")
else:
    print("\nNo missing values found.")

# Feature Scaling: Scale the feature columns (typically don't scale the target for regression)
scaler = StandardScaler()
data[['sqft', 'bedrooms']] = scaler.fit_transform(data[['sqft', 'bedrooms']])
print("\nFeature scaling applied to 'sqft' and 'bedrooms'.")

# Save the preprocessed data to a new CSV file
output_path = "C:\\dat1\\housing_preprocessed.csv"
data.to_csv(output_path, index=False)
print(f"\nPreprocessed data saved to {output_path}")
