# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:39:21 2017

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

def doPCA(input):
 #standardizing the data
 mean_of_each_column = np.mean(myData, axis=0)
 #print mean_of_each_column

 meanCenteredData = myData - mean_of_each_column
 print "--------------------------- Mean Centered Data ----------------------------------"
 print meanCenteredData

 x = myData[:,[0]]
 #print x
 y = myData[:,[1]]
 y_theoretical = myData[:,[2]]

 xMeanCentered = meanCenteredData[:,[0]]
 #print xMeanCentered
 yMeanCentered = meanCenteredData[:,[1]]
 #print yMeanCentered
 
 xy_centered_matrix = np.concatenate([xMeanCentered, yMeanCentered], axis=1)
 print "------------------------ final matrix(taking only x and y) -----------------------"
 print xy_centered_matrix
 #print np.shape(xy_centered_matrix)
 #a = xy_centered_matrix.T
 #print np.shape(a)

 covarMatrix = np.cov(xy_centered_matrix.T)
 #print covarMatrix

 eigen_values, eigen_vectors = LA.eig(covarMatrix)
 print "--------------------------- Eigen Values : ----------------------------------------"
 print eigen_values
 print "----------------------------- Eigen Vectors : --------------------------------------"
 print eigen_vectors

 #sorting the eigen values in descending order
 II = eigen_values.argsort()[::-1]
 eigen_values = eigen_values[II]
 eigen_vectors = eigen_vectors[:,II] #not reducing the dimension, considring all pcs
 print "------------------------- Eigen Values in Descending Order : --------------------"
 print  eigen_values
 print "-------------- Eigen Vectors of corresponding sorted eigen values : ---------------"
 print eigen_vectors
 #print np.shape(eigen_vectors)
 
 
 pca_matrix = np.dot(eigen_vectors.T, xy_centered_matrix.T).T
 print "------------------------------- PC matrix ------------------------------------------"
 print pca_matrix
 print "--------------------------------------------------------------------------------------"
 pc1 = np.dot(eigen_vectors[0].T,  xy_centered_matrix.T).T
 print "PC1 : ", pc1
 pc2 = pca_matrix[:,1]
 print "PC2 : ", pc2
 print "------------------------------- Final Plot -----------------------------------------"
 

 
 fig = plt.figure()
 ax = fig.add_subplot(1,1,1)
 ax.set_title('raw data and PC axis')
 ax.scatter(x,y,color = 'r')
 ax.scatter(x,y_theoretical,color = 'b')
 #ax.scatter(pc1,pc2,color = 'black')
 ax.plot((0, -5*(eigen_vectors[0,0])), (0, -5*(eigen_vectors[1,0])), color='green', linewidth=3)
 ax.set_aspect('equal', 'box')
 fig.show()
 
doPCA(myData)