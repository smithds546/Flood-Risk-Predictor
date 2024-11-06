import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("your_dataset.csv")

# Extract features (X) and target variable (y)
X = data[["AREA", "LDP"]].values
y_actual = data["Index flood"].values

# Define the coefficients for the LINEST model
intercept_linest = 66.20979
coef1_linest = 0.2440053
coef2_linest = -1.80195

# Calculate predicted values using LINEST model
y_pred_linest = intercept_linest + coef1_linest * X[:, 0] + coef2_linest * X[:, 1]

# Calculate MSE for LINEST model
mse_linest = mean_squared_error(y_actual, y_pred_linest)

# Calculate R-squared (R²) Score for LINEST model
r2_linest = r2_score(y_actual, y_pred_linest)

# Calculate predicted values using MLP model (assuming you have already trained the MLP model)
# Replace y_pred_mlp with actual predictions from your MLP model
y_pred_mlp = ...  # Replace ... with actual predictions from your MLP model

# Calculate MSE for MLP model
mse_mlp = mean_squared_error(y_actual, y_pred_mlp)

# Calculate R-squared (R²) Score for MLP model
r2_mlp = r2_score(y_actual, y_pred_mlp)

# Visual Comparison: Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_actual, y_pred_linest, color='blue', label='LINEST Predictions')
plt.scatter(y_actual, y_pred_mlp, color='red', label='MLP Predictions')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='black', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

print("LINEST Model:")
print("Mean Squared Error (MSE):", mse_linest)
print("R-squared (R²) Score:", r2_linest)
print("\nMLP Model:")
print("Mean Squared Error (MSE):", mse_mlp)
print("R-squared (R²) Score:", r2_mlp)
