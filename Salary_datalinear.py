# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:23:10 2023

@author: lavan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# Data
years_experience = np.array([101,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5])
salary = np.array([39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872])

# Reshape the data
years_experience = years_experience.reshape(-1,1)
salary = salary.reshape(-1,1)

# Split the data into train and test
X_train,X_test,Y_train,Y_test = train_test_split(years_experience,salary,test_size=0.2,random_state=42)

# Linear regression model
LR = LinearRegression()

# Fit the model to the training data
LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)
print("coffients:",LR.coef_)
print("Intercept:",LR.intercept_)
 
# Evaluate the model
mse = mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test,Y_pred)

print("Mean_squared_error:",mse)
print("Root Mean Squared Error:",rmse)
print("R-squared:",r2)

# Plot the regression line
plt.scatter(X_test,Y_test,color="black")
plt.plot(X_test,Y_pred,color="blue",linewidth=3)
plt.xlabel("Years of Experience")
plt.title("Linear Regression Model")
plt.ylabel("Salary")
plt.show()