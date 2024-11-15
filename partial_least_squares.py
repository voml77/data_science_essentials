
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data with multi collinearity
np.random.seed(42)
n_samples = 100
n_features = 10

# Create independent variables with multi collinearity
X = np.random.rand(n_samples, n_features)
# Introduce multi collinearity by correlating some features
for i in range(1, n_features):
    X[:, i] = X[:, i-1] + np.random.normal(0, 0.1, n_samples)

# Create a dependent variable with some non-linear relationship
# The coefficients randomly selected for the impact on y
coefficients = np.random.randn(n_features)
y = X @ coefficients + np.random.normal(0, 0.5, n_samples)  # Linear combination with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Partial Least Squares model
n_components = 5  # Number of components to use
pls = PLSRegression(n_components=n_components)
pls.fit(X_train, y_train)

# Make predictions
y_pred = pls.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Ideal Prediction')
plt.title('Partial Least Squares Regression Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
