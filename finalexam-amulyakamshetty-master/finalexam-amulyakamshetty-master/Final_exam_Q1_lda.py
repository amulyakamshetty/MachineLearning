# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:29:44 2017

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


#PLOTTING    
print('\n Projection Matrix: Y = X . W : \n')
X_lda = new_data.dot(W)
print X_lda
#print np.shape(X_lda)
     #assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
     
#def plot_step_lda():

    
    
    
#PLOTTING 
print "-------------------------------- FINAL PLOT---------------------------------------"   
ax = plt.subplot(111)
#for my_data,marker,color in zip((class_one, class_two),('^', 's', 'o'),('blue', 'red', 'green')):
    
plt.scatter(x = X_lda[0:30,[0]], y = X_lda[0:30,[1]], color = 'red', label = 'v1')
plt.scatter(x = X_lda[30:60,[0]], y = X_lda[0:30,[1]], color = 'BLUE', label = 'v2')   
#plt.scatter(x=X_lda[:,0], y = X_lda[:,1],color='red',alpha=0.5)
                #label=label_dict[label]
                

plt.xlabel('LD1')
plt.ylabel('LD2')

leg = plt.legend(loc='upper right', fancybox=True)
    # leg.get_frame().set_alpha(0.5)
plt.title('LDA: data projection onto the first 2 linear discriminants ( 2-D data )')

plt.grid()
plt.tight_layout
plt.show()
    

#1D plot
ax = plt.subplot(111)
plt.title("LDA plot")
plt.scatter(x = X_lda[0:30,[0]], y = np.zeros(30), color = 'red', label = 'v1')
plt.scatter(x = X_lda[30:60,[0]], y = np.zeros(30), color = 'BLUE', label = 'v2')
plt.legend(loc= 2)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.grid()
plt.show()



ax = plt.subplot(111)
plt.title("W matrix ")
ax.plot((0, -5*(W[0][0])), (0, -5*(W[1][0])), color='b', linewidth=3)
#plt.scatter(x = W[30:60,[0]], y = np.zeros(30), color = 'BLUE', label = 'v2')
plt.legend(loc= 2)
plt.xlabel('W')

plt.grid()
plt.show()

     