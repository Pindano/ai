import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perfomance import mean_squared_error, gradient_descent

# Load the dataset
data = pd.read_csv("Nairobi Office Price Ex.csv")
X = data['SIZE'].values  # Feature: office size
y = data['PRICE'].values  # Target: office price

# Initialize parameters for linear regression
np.random.seed(0)
m = -2  # Random initial slope
c = 90  # Random initial intercept
learning_rate = 0.001  # Small learning rate to ensure convergence
epochs = 10  # Number of iterations

# Training process
errors = []
for epoch in range(epochs):
    # Predict the target variable
    y_pred = m * X + c
    # Calculate and store the Mean Squared Error
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    print(f"Epoch {epoch + 1}, Mean Squared Error: {error}")
    # Update the slope and intercept using gradient descent
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plotting the line of best fit after training
plt.plot(X, y, color='blue', label="Data points")
plt.plot(X, m * X + c, color='red', label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("Office Size vs Price with Line of Best Fit")
plt.legend()
plt.show()

# Predict the office price when the size is 100 sq. ft.
predicted_price = m * 100 + c
print(f"Predicted office price for 100 sq. ft.: {predicted_price}")
