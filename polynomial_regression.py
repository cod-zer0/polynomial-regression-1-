# Polynomial Regression

# Data prepprocessing
# Importing  Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

# Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X.reshape(-1, 1),y)

# Fitting polinomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
regressor2 = PolynomialFeatures(degree = 5)
X_poli = regressor2.fit_transform(X.reshape(-1, 1))
regressor3 = LinearRegression()
regressor3.fit(X_poli, y)

# Visulizing the linear regression model
plt.scatter(X, y, color='red')
plt.plot(X,regressor.predict(X.reshape(-1, 1)),color ='green')
plt.title("Truth or Bluff")
plt.ylabel("salary")
plt.xlabel("position levell")

# Visulizing the polinomial regression model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1 ))
plt.scatter(X, y, color='red') 
plt.plot(X_grid,regressor3.predict(regressor2.fit_transform(X_grid.reshape(-1, 1))),color ='yellow')
plt.title("Truth or Bluff")
plt.ylabel("salary")
plt.xlabel("position levell")


# Pedicting result with linear regression
print(regressor.predict(np.array(6.5).reshape(1, -1)))


# Pedicting result with linear regression
a=np.array(6.5).reshape(1, -1)
print(regressor3.predict(regressor2.fit_transform(a)))











  
