# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('C://Users/Prani/Desktop/python/salary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
               
#Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.xlabel('Years of exp')
plt.ylabel('salary')
plt.title('Salary Vs Exp')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.xlabel('Years of exp')
plt.ylabel('salary')
plt.title('Salary Vs Exp')
plt.show()