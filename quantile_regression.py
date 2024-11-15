
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Simulate synthetic data
np.random.seed(42)

# Generate household income data (in thousands)
income = np.random.uniform(20, 120, 200)  # Income between $20,000 and $120,000

# Create an underlying relationship with some noise
# Food expenditures vary with income, but with some variability
expenditure = 0.3 * income + 5 + np.random.normal(0, 4, size=income.shape)

# Create a DataFrame
data = pd.DataFrame({
    'Income': income,
    'Expenditure': expenditure
})

# Perform Quantile Regression
quantiles = [0.25, 0.5, 0.75]  # 25th, 50th (median), and 75th percentiles
models = {}

# Fit quantile regression models for specified quantiles
for q in quantiles:
    model = smf.quantreg('Expenditure ~ Income', data)
    results = model.fit(q=q)
    models[q] = results

# Generate predictions for plotting
x_vals = np.linspace(20, 120, 100)
predictions = {q: model.predict({'Income': x_vals}) for q, model in models.items()}

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data['Income'], data['Expenditure'], alpha=0.5, label='Data', color='grey')
plt.plot(x_vals, predictions[0.25], color='blue', label='25th Quantile', linestyle='--')
plt.plot(x_vals, predictions[0.5], color='orange', label='50th Quantile (Median)', linestyle='--')
plt.plot(x_vals, predictions[0.75], color='green', label='75th Quantile', linestyle='--')
plt.title('Quantile Regression: Household Income vs Food Expenditures')
plt.xlabel('Household Income (in thousands)')
plt.ylabel('Food Expenditures (in thousands)')
plt.legend()
plt.grid()
plt.show()
