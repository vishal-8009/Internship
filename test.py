print("hi")
from pyexpat import model

import numpy as np
print(np.__version__)
import pandas as pd

# Load the dataset
data = pd.read_csv("bp.csv")

# Display the first 5 rows
print(data.head())

# Define target (y)
y = data["Systolic BP (mm Hg)"]

# Define features (x)
x = data.drop(columns=["Systolic BP (mm Hg)", "Subject_ID"])

# Check shapes
print("X shape:", x.shape)
print("Y shape:", y.shape)

# Split into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set:", x_train.shape)
print("Testing set:", x_test.shape)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, model.predict(x_test))
print{"Mean Squared Error:", error}