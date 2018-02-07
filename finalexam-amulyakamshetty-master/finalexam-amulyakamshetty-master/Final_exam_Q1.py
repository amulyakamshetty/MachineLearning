# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from numpy import genfromtxt



#reading the data from csv file and omitting the first row and column   
my_data = np.loadtxt('C:\Users\Amulya\Desktop\ML_codes\dataset_1 (1).csv',skiprows=1,usecols=range(0,3),delimiter=',')
print ("Data present in the .csv file: \n")
print my_data
#print np.shape(my_data)

def doPCA(input):
 #standardizing the data
 #mean_of_each_column = np.mean(my_data, axis=0)
 #print mean_of_each_column

 #meanCenteredData = my_data - mean_of_each_column
 #print "--------------------------- Mean Centered Data ----------------------------------"
 #print meanCenteredData
 
 v1 = my_data[:,0]
 v1 = np.reshape(v1, (60,1))
 v2 = my_data[:,1]
 v2 = np.reshape(v2, (60,1))
 label = my_data[:,2]
 label = np.reshape(label, (60,1))
 new_data = np.concatenate((v1, v2), axis = 1)
 
 
 #finding the covariance matrix
 cov_matrix = np.cov(new_data.T)
 
 #finding out the eigen values and eigen vectors
 eigen_val, eigen_vec = LA.eig(cov_matrix)
 
 print "--------------------------- Eigen Values : ----------------------------------------"
 print eigen_val
 print "----------------------------- Eigen Vectors : --------------------------------------"
 print eigen_vec
 
 #sorting the eigen values
 II = eigen_val.argsort()[::-1]
 eigen_val = eigen_val[II]
 eigen_vec = eigen_vec[:,II] #not reducing the dimension, considring all pcs
 print "------------------------- Eigen Values in Descending Order : --------------------"
 print  eigen_val
 print "-------------- Eigen Vectors of corresponding sorted eigen values : ---------------"
 print eigen_vec
 
 print('\n Variance explained:\n')
 eigv_sum = np.sum(eigen_val)
 for i,j in enumerate(eigen_vec):
     print('eigenvalue {0:}: {1:.2%}'.format(i+1, ((j[0]/eigv_sum)*-1*100).real))
 
 #PCA matrix
 PCA_matrix = np.dot((eigen_vec.T),  (new_data.T)).T
 print "--------------------------- PCA Matric : ----------------------------------------"
 print PCA_matrix
 
 #PC_1 = np.dot((eigen_vec[0].T),(new_data).T) 
 #PC1 = PC_1.T
 
 #PC_1 = PCA_matrix[:,0]
 #PC_2 = PCA_matrix[:,1]
 
 print "--------------------------- Projection Matrix ----------------------------------------"
 y = np.dot((PCA_matrix[:,0]).T, new_data)
 print y
 
 
 #Plotting 
 
 fig = plt.figure(1)
 ax = fig.add_subplot(1,1,1)
 ax.set_title('Plot 1. raw data - v1 vs v2')
 ax.scatter(v1,v2,color = 'red', label ='v1 vs v2')
 #ax.scatter(pc1,pc2,color = 'black')
 #ax.plot((0, -5*(eigen_vec[0,0])), (0, -5*(eigen_vec[1,0])), color='blue', linewidth=3)
 #ax.plot((0, -5*(y[0])), (0, -5*(y[1])), color='yellow', linewidth=3)
 
 ax.set_aspect('equal', 'box')
 fig.show()
 
 
 fig = plt.figure(2)
 ax = fig.add_subplot(1,1,1)
 ax.set_title('2. projection onto PC1 axis')
 #ax.scatter(v1,v2,color = 'red')
 #ax.scatter(pc1,pc2,color = 'black')
 #ax.plot((0, -5*(eigen_vec[0,0])), (0, -5*(eigen_vec[1,0])), color='blue', linewidth=3)
 ax.plot((0, -5*(y[0])), (0, -5*(y[1])), color='b', linewidth=3)
 
 fig = plt.figure(3)
 ax = fig.add_subplot(1,1,1)
 ax.set_title('3. raw data and PC axis')
 ax.scatter(v1,v2,color = 'red')
 #ax.scatter(pc1,pc2,color = 'black')
 #ax.plot((0, -5*(eigen_vec[0,0])), (0, -5*(eigen_vec[1,0])), color='blue', linewidth=3)
 ax.plot((0, -5*(y[0]/4500)), (0, -5*(y[1]/4500)), color='y', linewidth=3)
 
 ax.set_aspect('equal', 'box')
 fig.show()
 
 ax.set_aspect('equal', 'box')
 fig.show()
 
doPCA(my_data)




