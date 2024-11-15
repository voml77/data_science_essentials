
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate some sample data
# X values (features) - using a simple range
X = np.sort(5 * np.random.rand(80, 1), axis=0)
# y values (targets) - using a function with some noise
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Add some noise

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the data
rf_regressor.fit(X, y)

# Predict using the model
X_pred = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = rf_regressor.predict(X_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y, rf_regressor.predict(X)))
print(f'Root Mean Square Error (RMSE): {rmse:.2f}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(X_pred, y_pred, color='blue', label='Random Forest Prediction')
plt.title('Random Forest Regression')
plt.xlabel('X values')
plt.ylabel('y values')
plt.legend()
plt.grid()
plt.show()
