from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Simulated data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Initial Mean Squared Error: {mse}")

# Example of adaptation: Retrain model with new data
new_data = np.array([[6], [7]])
new_target = np.array([12, 14])
X_train_extended = np.vstack([X_train, new_data])
y_train_extended = np.concatenate([y_train, new_target])
model.fit(X_train_extended, y_train_extended)

# Predict with the updated model
updated_predictions = model.predict(X_test)
updated_mse = mean_squared_error(y_test, updated_predictions)
print(f"Updated Mean Squared Error: {updated_mse}")
