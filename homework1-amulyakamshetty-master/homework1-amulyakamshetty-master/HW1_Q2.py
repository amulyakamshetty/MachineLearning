# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:02:18 2017

@author: Amulya
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from numpy import genfromtxt
#from sklearn.decomposition import PCA
from matplotlib.mlab import PCA



#reading the data from csv file and omitting the first row and column   
myData = np.loadtxt('C:\Users\Amulya\Downloads\linear_regression_test_data.csv',skiprows=1,usecols=range(1,4), delimiter=',')
print ("Data present in the .csv file : ")
print  myData

def do_linear_regression(input) :
 x = myData[:,[0]]
 #print x
 one_matrix = np.ones((20,1))
 #print one_matrix
 new_x = np.hstack((x, one_matrix))
 print "---------------------------------- FINAL x -------------------------------------- "
 print new_x
 
 #print x
 y = myData[:,[1]]
 print "--------------------------------------- Y ----------------------------------------"
 print y
 y_theoretical = myData[:,[2]]
 
 x_transpose = new_x.T
 a = np.dot(x_transpose, new_x)
 #print a 
 #print np.shape(a)
 a_inv = LA.inv(a)
 #print a_inv
 
 b = np.dot(x_transpose, y)
 #print b
 print "------------------------------- (beta[0], beta[1]) -----------------------------------"
 beta = np.dot(a_inv, b)
 print beta
 #print beta[1]
 print "------------------------------- y = beta[1] + beta[0]x -------------------------------"
 #y_reg = np.dot(x_reg, beta.T)
 #print y_reg
 
 yhat = beta[1] + beta[0]*x
 print yhat
 
 
 print "-------------------------------- FINAL PLOT ------------------------------------------"
 fig = plt.figure()
 ax = fig.add_subplot(1,1,1)
 ax.set_title('regression line')
 ax.scatter(x,y,color = 'red')
 ax.scatter(x,y_theoretical,color = 'blue')
 #ax.scatter(x, yhat, color='green', linewidth=3)
 plt.plot(x, yhat, color='green', linewidth=3)
 
 
 ax.set_aspect('equal', 'box')
 fig.show()
 
do_linear_regression(myData) 