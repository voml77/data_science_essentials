
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic non-linear data
np.random.seed(42)
X = np.sort(10 * np.random.rand(100, 1), axis=0)  # 100 samples, feature scaled from 0 to 10
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Non-linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform features to polynomial features
degree = 4  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Create and fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Make predictions
y_pred = model.predict(X_poly_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data', alpha=0.6)
plt.scatter(X_test, y_test, color='red', label='Test Data', alpha=0.6)
X_poly_full = poly_features.transform(X)  # Transform the entire dataset for plotting
y_poly_pred = model.predict(X_poly_full)
plt.plot(X, y_poly_pred, color='orange', label='Polynomial Regression Fit', linewidth=2)
plt.title('Polynomial Regression for Non-Linear Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
