import numpy as np


# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error between true and predicted values.

    Parameters:
    y_true (array-like): Actual values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    float: Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)


# Function to perform one step of Gradient Descent for Linear Regression
def gradient_descent(X, y, m, c, learning_rate):
    """
    Perform one step of gradient descent and update the slope and intercept values.

    Parameters:
    X (array-like): Input feature values (office sizes).
    y (array-like): Target values (office prices).
    m (float): Current slope.
    c (float): Current intercept.
    learning_rate (float): The learning rate for gradient descent.

    Returns:
    tuple: Updated values of slope (m) and intercept (c).
    """
    N = len(y)
    y_pred = m * X + c  # Predicted y values based on current m and c
    # Calculate gradients for m and c
    dm = -(2 / N) * np.sum(X * (y - y_pred))
    dc = -(2 / N) * np.sum(y - y_pred)
    # Update m and c
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c
