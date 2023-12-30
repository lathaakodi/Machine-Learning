# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:58:58 2023

@author: lavan
"""
# Importing the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv("C://Users//lavan//Downloads//50_Startups.csv")
dataset.head() 

# Split the x and Y variable
X = dataset.iloc[:,:-1]
X
y = dataset.iloc[:, 4]
y

# Convert categorical variable
states=pd.get_dummies(X['State'],drop_first=True)
states
X =X.drop("State",axis=1)
X
X = pd.concat([X,states],axis=1)
X 

# Standardization
SS = StandardScaler()
SS_X = SS.fit_transform(X)
pd.DataFrame(SS_X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0) 

# Fit the model
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(X_train,y_train)

# Y predictions
y_pred = LR.predict(X_test)
y_pred
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score 

# Model Evaluation
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2 Value: {r2}')

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X_with_constant = add_constant(X_train)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]
print("\nVIF Values:")
print(vif_data)

# Cook's Distance
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
influence = model.get_influence()
cook_distance = influence.cooks_distance[0]

# Plot Cook's Distance
plt.figure(figsize=(12, 6))
plt.stem(cook_distance, markerfmt=",", basefmt=" ", linefmt="r-.")
plt.title("Cook's Distance Plot")
plt.xlabel("Data Points")
plt.ylabel("Cook's Distance")
plt.show()

# Identify influential points (e.g., points with Cook's distance > 4/n)
influential_points = np.where(cook_distance > 4 / X_train.shape[0])[0]
print("\nInfluential Points:", influential_points)

# Model Deletion Diagnostics
model_summary = model.summary()
print(model_summary)

# Residuals vs Predicted Values
residuals = model.resid
plt.figure(figsize=(10, 6))
plt.scatter(model.predict(X_train), residuals)
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()






