# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 06:28:31 2017

@author: Amulya
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Loading the diabetes dataset
diabetes = datasets.load_diabetes()
print "------------------------------ data present in the array ------------------------------"
print diabetes

#print np.shape(diabetes.data)
#print np.shape(diabetes.target)
# Use only one feature
x = diabetes.data[:, np.newaxis, 2]

# Splitting the data into training and testing sets
x_training = x[:-20]
x_testing = x[-20:]

#print np.shape(x_training)
#print np.shape(x_testing)

# Splitting the targets into training and testing sets
y_training = diabetes.target[:-20]
y_testing = diabetes.target[-20:]

# Creating linear regression object
regression_obj = linear_model.LinearRegression()

# Training the regression model using the training sets
regression_obj.fit(x_training, y_training)

# Making predictions using the testing set
y_prediction = regression_obj.predict(x_testing)

# printing out the coefficients
#print('Coefficients: ', regression_obj.coef_)
# printing out the mean squared error
#print("Mean squared error: %.2f"
 #     % mean_squared_error(y_testing, y_prediction))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_testing, y_prediction))

print "-------------------------------- FINAL PLOT---------------------------------------"

# Plotting the final plots
plt.title('regression line')
plt.xlabel('testing x')
plt.ylabel('y (testing/predicted)')
plt.scatter(x_testing, y_testing,  color='green')
plt.plot(x_testing, y_prediction, color='red', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()