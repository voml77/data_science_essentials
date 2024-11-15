
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the data
train_file = 'data/train.csv'
test_file = 'data/test.csv'

# Read the CSV files
train_data = pd.read_csv(train_file, delimiter=';')
test_data = pd.read_csv(test_file, delimiter=';')


# Prepare the data
X_train = train_data[['Height']]
y_train = train_data['Weight']
X_test = test_data[['Height']]
y_test = test_data['Weight']

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# coefficients of the trained model
print('\nCoefficient of model :', model.coef_)

# intercept of the model
print('\nIntercept of model', model.intercept_)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Print predictions
print("Predictions for the test data:")
for height, weight in zip(X_test['Height'], predictions):
    print(f'Height: {height}, Predicted Weight: {weight:.2f}')

# Optional: Plotting results
plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0)
plt.scatter(X_test, y_test, color='red', label='Test data', alpha=0.5)
plt.plot(X_test, predictions, color='green', label='Regression line')

#  Highlight a specific independent variable point
# For example, highlight the point where Height = 182
highlight_height = 182
# Predict weight for this height
highlight_weight = model.predict(pd.DataFrame([[highlight_height]], columns=['Height']))[0]

plt.scatter(highlight_height, highlight_weight, color='yellow', edgecolor='black',
            label=f'Highlighted point (Height: {highlight_height})', s=200)

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression: Height vs. Weight')
plt.legend()
plt.show()
