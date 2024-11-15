
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Simulate synthetic data for drug dosage and blood pressure
np.random.seed(42)

# Generate drug dosage data (0 to 100 mg)
dosage = np.linspace(0, 100, 50)
# Create an underlying linear relationship with some noise
true_slope = -0.5   # Assume higher dosage leads to lower blood pressure
true_intercept = 120  # Average blood pressure when dosage is zero
true_sigma = 5.0     # Standard deviation of the noise

# Simulate blood pressure values
blood_pressure = true_intercept + true_slope * dosage + np.random.normal(0, true_sigma, size=dosage.shape)

# Adding a bias term (intercept) to dosage
X_matrix = np.column_stack((np.ones(dosage.shape[0]), dosage))

# Prior parameters
alpha_prior = 0  # Prior for intercept
beta_prior = 0   # Prior for slope
sigma_prior = 10  # Larger prior uncertainty

# Calculate posterior means
n = len(blood_pressure)
XTX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
beta_post = XTX_inv @ (X_matrix.T @ blood_pressure)

# Posterior mean
posterior_mean = beta_post

# Posterior covariance
posterior_var = true_sigma**2 * XTX_inv

# Generate predictions
x_test = np.linspace(0, 100, 100)
X_test_matrix = np.column_stack((np.ones(x_test.shape[0]), x_test))
y_pred = X_test_matrix @ posterior_mean

# Correctly calculate the predictive standard deviation
y_pred_var = np.diag(X_test_matrix @ posterior_var @ X_test_matrix.T) + true_sigma**2
y_pred_std = np.sqrt(y_pred_var)

# Calculate credible intervals
lower_bound = y_pred - 1.96 * y_pred_std
upper_bound = y_pred + 1.96 * y_pred_std

# Calculate RMSE using the modeled predictions vs actual values
# Use the blood pressure values corresponding to the dosage values for RMSE calculation
# Since the model predicts for continuous values, let's use predictions
# at random dosage points (or directly from blood pressure)
y_pred_for_rmse = X_matrix @ posterior_mean  # Predictions based on the training data

# Compute RMSE
rmse = np.sqrt(mean_squared_error(blood_pressure, y_pred_for_rmse))
print(f'Root Mean Square Error (RMSE): {rmse:.2f}')

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(dosage, blood_pressure, label='Data', color='blue', alpha=0.6)
plt.plot(x_test, y_pred, color='orange', label='Posterior Predictive Mean', linewidth=2)
plt.fill_between(x_test, lower_bound, upper_bound, color='orange', alpha=0.3,
                 label='95% Credible Interval')
plt.title('Bayesian Linear Regression: Drug Dosage vs Blood Pressure')
plt.xlabel('Drug Dosage (mg)')
plt.ylabel('Blood Pressure (mm Hg)')
plt.legend()
plt.grid()
plt.show()
