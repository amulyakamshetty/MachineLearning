# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 01:51:37 2017

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
 

 
 
 #LDA
 
 #data has 2 features and 2 classes
 v1 = my_data[:,0]
 v1 = np.reshape(v1, (60,1))
 v2 = my_data[:,1]
 v2 = np.reshape(v2, (60,1))
 label = my_data[:,2]
 label = np.reshape(label, (60,1))
 new_data = np.concatenate((v1, v2), axis = 1)

 print "-------------------------CLASS ONE------------------------------------------"
 class_one = new_data[0:30,:]
 print class_one
 #print np.shape(class_one)

 print "-------------------------CLASS TWO------------------------------------------"
 class_two = new_data[30:,:]
 print class_two
 #print np.shape(class_two)

 class_one_mean = class_one.mean(axis=0)
 print "------------------------- Mean Vector Class 1 : -------------------------"
 print class_one_mean


 print "------------------------- Mean Vector Class 2 : -------------------------"
 class_two_mean = class_two.mean(axis=0)
 print class_two_mean

 #SCATTER-WITHIN MATRIX CALCULATION

 S_1 = np.zeros((2,2))                  # scatter matrix for every class
 for row in class_one:
        row = row.reshape(2,1) # make column vectors
        #print "hhhhhhhhhhhhhhhhh" , row
        class_one_mean = class_one_mean.reshape(2,1)
        S_1 = S_1 + (row-class_one_mean).dot((row-class_one_mean).T)
 print S_1

 S_2 = np.zeros((2,2))  
 for row in class_two:
        row = row.reshape(2,1) # make column vectors
        #print "hhhhhhhhhhhhhhhhh" , row
        class_two_mean = class_two_mean.reshape(2,1)
        S_2 = S_2 + (row - class_two_mean).dot((row-class_two_mean).T)
 print S_2

 S_W = S_1 + S_2
 print "-------------------------------- Scatter within matrix -------------------------------"
 print S_W


 #calculating between class scatter matrix 
 overall_mean = np.mean(new_data, axis = 0)
 #overall_mean1 =  (class_one_mean +  class_two_mean)/2  
 overall_mean = overall_mean.reshape(2,1) 
 print overall_mean
 #print overall_mean1


 class_one_mean_reshape = class_one_mean.reshape(2,1)
   #print np.shape( class_one_mean_reshape)
 class_two_mean_reshape = class_two_mean.reshape(2,1)
   #print np.shape(class_one_mean )
 c = (class_one_mean_reshape - overall_mean )
#print c
#print np.shape(c)


 S_B_1 = 30 * c.dot(c.T)
#print S_B_1
#print np.shape(S_B_1)
   
 S_B_2 = (30 * (class_two_mean_reshape - overall_mean ).dot((class_two_mean_reshape - overall_mean ).T))

 S_B = S_B_1 + S_B_2
 print "------------------------- scatter between matrix -------------------------------"
 print S_B
 print np.shape(S_B)


 S_W_inv = LA.inv(S_W)
 #print S_W_inv
   
 final_matrix = (S_W_inv).dot(S_B)
 print "------------------------------- final  matrix ---------------------------------"
 print final_matrix 


 eig_vals, eig_vecs = LA.eig(final_matrix)
   
 for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(2,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print "\n"
 print('Eigenvalues in decreasing order:\n')
 for i in eig_pairs:
     print(i[0])
     


 print('\n Variance explained:\n')
 eigv_sum = np.sum(eig_vals)
 for i,j in enumerate(eig_pairs):
     print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
     #choosing maximum eigen values 
 W = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
 print('\n Matrix W:\n')
 print W.real
#print np.shape(W)     


 print('\n Projection Matrix: Y = X . W : \n')
 X_lda = new_data.dot(W)
 print X_lda
#print np.shape(X_lda)
     #assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
     
#def plot_step_lda():

    
    
    
#PLOTTING 

 print "-------------------------------- FINAL PLOT---------------------------------------"    

 fig = plt.figure(1)
 ax = fig.add_subplot(1,1,1)
 ax.set_title('6. raw data and PC axis and W axis')
 ax.scatter(v1,v2,color = 'red',  label = 'raw data')
#ax.scatter(pc1,pc2,color = 'black')
#ax.plot((0, -5*(eigen_vec[0,0])), (0, -5*(eigen_vec[1,0])), color='blue', linewidth=3)
 ax.plot((0, -5*(y[0]/4000)), (0, -5*(y[1]/4000)), color='green', linewidth=3,  label = 'PC1')
 
 #ax.set_aspect('equal', 'box')
 #fig.show()

 #ax.set_aspect('equal', 'box')
 #fig.show()
 


 ax = plt.subplot(111)
 #plt.title("W matrix ")
 ax.plot((0, -5*(W[0][0])*5), (0, -5*(W[1][0])*5), color='b', linewidth=3,  label = 'W axis')
#plt.scatter(x = W[30:60,[0]], y = np.zeros(30), color = 'BLUE', label = 'v2')
 plt.legend(loc= 2)
 plt.xlabel('W')  
 
 
 
#1D plot
 #ax = plt.subplot(121)
 #plt.title("LDA plot")
 #ax.scatter(x = X_lda[0:30,[0]]*10, y = np.zeros(30), color = 'red', label = 'v1')
 #ax.scatter(x = X_lda[30:60,[0]]*10, y = np.zeros(30), color = 'BLUE', label = 'v2')
 #ax.legend(loc= 2)
 #ax.xlabel('LD1')
 #ax.ylabel('LD2')
 #ax.grid()
 fig.show()
 
doPCA(my_data) 