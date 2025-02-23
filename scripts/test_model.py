# test.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:\\dat1\\housing.csv")

# Separate features and target variable
X = data.drop("price", axis=1)
y = data["price"]

# Split the data (using the same test_size and random_state as in train.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the trained model from file
with open("C:\\dat1\\models\\model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Test set predictions:", predictions)
print("Actual values:", y_test.values)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
