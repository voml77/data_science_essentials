
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
n_samples = 100
n_features = 5

# Create highly correlated variables
X = np.random.rand(n_samples, n_features)
# Introduce correlations
X[:, 1] = X[:, 0] + np.random.normal(0, 0.01, n_samples)
X[:, 2] = X[:, 0] + np.random.normal(0, 0.01, n_samples)
X[:, 3] = X[:, 0] + np.random.normal(0, 0.01, n_samples)
X[:, 4] = X[:, 0] + np.random.normal(0, 0.01, n_samples)

# Create response variable with some noise
true_coefficients = np.array([1.5, -2.0, 1.0, 0.5, 0.0])
y = X @ true_coefficients + np.random.normal(0, 0.1, n_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Determine the number of components to keep (for this example we keep 3)
num_components = 3
X_train_pca_reduced = X_train_pca[:, :num_components]
X_test_pca_reduced = X_test_pca[:, :num_components]

# Fit a linear regression model using the principal components
model = LinearRegression()
model.fit(X_train_pca_reduced, y_train)

# Predictions
y_pred = model.predict(X_test_pca_reduced)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.4f}')

# Plotting - First we will compare the predicted values against the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Principal Component Regression: Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plotting the explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_features + 1), explained_variance, alpha=0.7, align='center')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.xticks(range(1, n_features + 1))
plt.grid()
plt.show()
