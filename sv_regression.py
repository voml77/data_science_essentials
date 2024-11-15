
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Generate some sample data
# X values (features) - using a simple range
X = np.sort(5 * np.random.rand(80, 1), axis=0)
# y values (targets) - using a function with some noise
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Add some noise

# Create the SVR model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

# Fit the model to the data
svr_rbf.fit(X, y)

# Predict using the model
X_pred = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = svr_rbf.predict(X_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, svr_rbf.predict(X)))
print(f'Root Mean Square Error (RMSE): {rmse:.2f}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X_pred, y_pred, color='blue', label='SVR Prediction')
plt.title('Support Vector Regression (SVR)')
plt.xlabel('X values')
plt.ylabel('y values')
plt.legend()
plt.grid()
plt.show()
