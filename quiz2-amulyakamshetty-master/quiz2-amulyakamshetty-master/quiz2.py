# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:50:41 2017

@author: Amulya
"""

# -*- coding: utf-8 -*-





import numpy as np

from numpy import linalg as la

import matplotlib.pyplot as plt



my_data = np.loadtxt('G:/UNCC/Subjects/ML/Q-2/SCLC_study_output_filtered.csv',skiprows=1,usecols=range(1,50),delimiter=',')

print ("Data present in the .csv file: \n",my_data)



def PCA(input):

    

    data_mean = np.mean(my_data,axis=0)

   # x,y,z = my_data.transpose()



    normalized_data = np.subtract(my_data,data_mean)

    normalized_data_trans = normalized_data.T

    

    #Covariance matrix

    covarmatrix = np.dot(normalized_data_trans,normalized_data)/len(normalized_data)

    variance = np.diagonal(covarmatrix)

    print ("Variance is:",variance)

    sum = np.sum(variance)

    print ("Total Variance:",sum)

    np.cov(my_data)

    print("Covariance Matrix is:", covarmatrix)

    

  #  scaled_data = np.column_stack([my_data[:,0]-mean_x,my_data[:,1]-mean_y,my_data[:,2]-mean_z])

   # scaled_data_trans = np.transpose(scaled_data)

    #covar_matrix = np.dot(scaled_data_trans,scaled_data)/len(x-1)

    #print ("----------------------")

    #print("Covariance Matrix: \n",covar_matrix)

    #print ("----------------------")



    #Finding Eigen values



    eigen_val, eigen_vector = la.eig(covarmatrix)

    print("Eigen Values: \n",eigen_val)

    print("Eigen Vector: \n",eigen_vector)



    #Finding Max Eigen Values



    eigen_max_1 = np.argmax(eigen_val)

    #variance_eigen_max_1 = np.var(eigen_max_1)

    #print("Variance of 1st PC:",variance_eigen_max_1)

    eigen_max_1_vector = (eigen_vector[:,eigen_max_1])

   # variance_eigen_max_1 = np.var(eigen_max_1_vector)

    #print("Variance of 1st PC:",variance_eigen_max_1)



    #deleting the max values

    eigen_new_val = np.delete(eigen_val,eigen_max_1)

    eigen_new_vector = np.delete(eigen_vector,eigen_max_1,1)



    #Finding the next max values

    eigen_max_2 = np.argmax(eigen_new_val)

    #variance_eigen_max_2 = np.var(eigen_max_2)

    #print("Variance of 2nd PC:",variance_eigen_max_2)

    eigen_max_2_vector = (eigen_new_vector[:,eigen_max_2])

    

   # total_pc_variance = np.add(variance_eigen_max_1,variance_eigen_max_2)

    #print("Total variance of PC:",total_pc_variance)



    print("PC1: \n",eigen_max_1_vector)

    print("PC2: \n",eigen_max_2_vector)



    feature_vector = np.column_stack([eigen_max_1_vector,eigen_max_2_vector])



    print ("----------------------")

    print("Feature Vector: \n",feature_vector)

    print ("----------------------")



    newdata = np.dot(feature_vector.T,normalized_data_trans)

    print("PCA DATA: \n",newdata)

    print ("----------------------")

    

    variance_PC = np.var(newdata)

    print("Variance of PC:",variance_PC)

    covariance_PC = np.cov(newdata)

    print("Co-variance of PC:",covariance_PC)



    fig = plt.figure()

    Ax = fig.add_subplot(1,1,1)

    Ax.scatter(newdata[0,:],newdata[1,:])

    fig.show()



PCA(my_data)