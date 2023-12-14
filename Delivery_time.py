# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:07:31 2023

@author: lavan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Input data
delivery_time = np.array([21, 13.5, 19.75 , 24, 29, 15.35, 19, 9.5, 17.9, 18.75, 19.83, 10.75, 16.68, 11.5, 12.03, 14.88, 13.75, 18.11, 8, 17.83, 21.5])
sorting_time = np.array([10, 4, 6, 9, 10, 6, 7, 3, 10, 9, 8, 4, 7, 3, 3, 4, 6, 7, 2, 7, 5])

# Reshape the data
delivery_time = delivery_time.reshape(-1, 1)
sorting_time = sorting_time.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sorting_time, delivery_time, test_size=0.2, random_state=42)

# Create and train the linear regression model
LR= LinearRegression()
LR.fit(X_train, y_train)

# Make predictions on the test set
y_pred = LR.predict(X_test)

# Print the model coefficients
print(f'Intercept: {LR.intercept_[0]}')
print(f'Coefficient: {LR.coef_[0][0]}')

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared (R2) Score: {r2}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression Model')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()
