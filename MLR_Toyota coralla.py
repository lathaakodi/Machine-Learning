# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:24:42 2023

@author: lavan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.regression import linear_model
# Read the CSV file
df = pd.read_csv("C://Users//lavan//Downloads//ToyotaCorolla (1).csv",encoding="latin1")

# Display the first few rows of the DataFrame
print(df.head())

# Select features and target variable
X = df[["Age_08_04", "Weight"]]
Y = df["Price"]

x_pred = np.linspace(6, 24, 30)     
y_pred = np.linspace(0.93, 2.9, 30)  
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T


model = LinearRegression()
model = model.fit(X,Y )
predicted = model.predict(model_viz)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)
model.fit(X,Y)

# Display the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict CO2 for the test data
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Model Validation Techniques

# Multicollinearity: Check variance inflation factor (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("VIF Results:")
print(vif_data)

# Cook's distance
influence = linear_model.OLS(Y, X).fit()
cooks_distance = influence.get_influence().cooks_distance[0]
print("Cook's Distance:")
print(cooks_distance)


# Model deletion diagnostics
fig, ax = plt.subplots(figsize=(12, 8))
influence_plot(influence, ax=ax)
plt.show()

# Residual analysis
residuals = Y_test - Y_pred
plt.figure(figsize=(8, 6))
plt.scatter(Y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
