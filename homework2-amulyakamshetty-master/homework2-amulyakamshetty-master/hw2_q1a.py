# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 10:43:49 2017

@author: Amulya
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Loading the diabetes dataset
#reading the data from csv file and omitting the first row and column   
my_data = np.loadtxt('C:\Users\Amulya\Downloads\SCLC_study_output_filtered_2.csv',skiprows=1,usecols=range(1,20),delimiter=',')
#print ("Data present in the .csv file: \n")
#print my_data





# Use only one feature


#USING SKLEARN LIBRARY
 
y1 = np.ones((20,1))
y2 = np.zeros((20,1))
y = np.concatenate((y1, y2), axis = 0)


# Creating linear discriminant object
clf = LinearDiscriminantAnalysis()
#clf.fit(my_data, y)

# Training the lda model using the training sets
x = clf.fit_transform(my_data, y)

print "-------------------------------- FINAL PLOT---------------------------------------"

plt.title("LDA plotted using SKLEARN")
plt.scatter(x = x[0:20,[0]], y = np.zeros((20)), color = 'green', label = 'NSCLC')
plt.scatter(x = x[20:40,[0]], y = np.zeros((20)), color = 'orange', label = 'SCLC')
plt.legend(loc= 2)
plt.grid()
plt.show()






# Training the lda model using the training sets
#clf.fit(x_training, y_training)

# Making predictions using the testing set
#y_prediction = clf.predict(x_testing)

# printing out the coefficients
#print('Coefficients: ', regression_obj.coef_)
# printing out the mean squared error
#print("Mean squared error: %.2f"
 #     % mean_squared_error(y_testing, y_prediction))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_testing, y_prediction))



# Plotting the final plots
#plt.title('LDA line')
#plt.xlabel('testing x')
#plt.ylabel('y')
#plt.scatter(x_testing, y_testing,  color='green')
#plt.plot(x_testing, y_prediction, color='red', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()