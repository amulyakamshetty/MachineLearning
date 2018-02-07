# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 10:43:49 2017

@author: Amulya
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from numpy import genfromtxt



#reading the data from csv file and omitting the first row and column   
my_data = np.loadtxt('C:\Users\Amulya\Downloads\SCLC_study_output_filtered_2.csv',skiprows=1,usecols=range(1,20),delimiter=',')
#print ("Data present in the .csv file: \n")
#print my_data


#print "-------------------------CLASS ONE------------------------------------------"
class_one = my_data[0:20,:]
#print class_one
#print np.shape(class_one)

#print "-------------------------CLASS TWO------------------------------------------"
class_two = my_data[20:,:]
#print class_two
#print np.shape(class_two)


#def d_LDA(x):
 
   #computing mean vectors 
class_one_mean = class_one.mean(axis=0)
#print "Mean Vector Class 1 : "
#print class_one_mean
#print np.shape(class_one_mean)
#print "\n Mean Vector Class 2 : "
class_two_mean = class_two.mean(axis=0)
#print class_two_mean
   #print np.shape(class_two_mean)
   
S_1 = np.zeros((19,19))                  # scatter matrix for every class
for row in class_one:
        row = row.reshape(19,1) # make column vectors
        #print "hhhhhhhhhhhhhhhhh" , row
        class_one_mean = class_one_mean.reshape(19,1)
        S_1 = S_1 + (row-class_one_mean).dot((row-class_one_mean).T)
   
   
   
   
  
   #a = row - class_one_mean
   #computing scatter matrices 
   #within class scatter
   #a = class_one - class_one_mean
   #print a 
   #print np.shape(a)
   #X_M1 = a.T
   #print X_M1
   #print np.shape(X_M1)
   #S_1 = (a).dot(a.T)
   #S_1 = ((X_M1).dot((X_M1).T)) 
#print S_1
   
   
S_2 = np.zeros((19,19))  
for row in class_two:
        row = row.reshape(19,1) # make column vectors
        #print "hhhhhhhhhhhhhhhhh" , row
        class_two_mean = class_two_mean.reshape(19,1)
        S_2 = S_2 + (row - class_two_mean).dot((row-class_two_mean).T)
#print np.shape(S_2)
   
   #class_two_mean =  class_two_mean.reshape(19,1)
   #for row in class_two:
    #       row = row.reshape(19,1)
   #print row   
   
   
   #b = row - class_two_mean
   #print b 
   #print np.shape(b)
   #X_M2 = b.T
   #print X_M2
   #print np.shape(X_M2)
   
   #S_2 = (b).dot(b.T)
   #S_2 = ((X_M2).dot((X_M2).T)) 
#print S_2
#print np.shape(S_2)
   
S_W = S_1 + S_2
print "-------------------------------- Scatter within matrix -------------------------------"
print S_W
  
#calculating between class scatter matrix 
overall_mean = np.mean(my_data, axis = 0)
   #overall_mean =  (class_one_mean +  class_two_mean)/2  
overall_mean = overall_mean.reshape(19,1)
print overall_mean
   
   #for i in range(1,3):
   #n = my_data [0, :].shape[0]
   #print n
class_one_mean_reshape = class_one_mean.reshape(19,1)
   #print np.shape( class_one_mean_reshape)
class_two_mean_reshape = class_two_mean.reshape(19,1)
   #print np.shape(class_one_mean )
c = (class_one_mean_reshape - overall_mean )
#print c
print np.shape(c)
S_B_1 = 20 * c.dot(c.T)
#print S_B_1
#print np.shape(S_B_1)
   
S_B_2 = (20 * (class_two_mean_reshape - overall_mean ).dot((class_two_mean_reshape - overall_mean ).T))
#print S_B_2
   
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
    eigvec_sc = eig_vecs[:,i].reshape(19,1)   
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
W = np.hstack((eig_pairs[0][1].reshape(19,1), eig_pairs[1][1].reshape(19,1)))
print('\n Matrix W:\n')
print W.real
#print np.shape(W)
     
     

     
 #PLOTTING    
print('\n Projection Matrix: Y = X . W : \n')
X_lda = my_data.dot(W)
print X_lda
#print np.shape(X_lda)
     #assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
     
#def plot_step_lda():

    
    
    
#PLOTTING 
print "-------------------------------- FINAL PLOT---------------------------------------"   
ax = plt.subplot(111)
#for my_data,marker,color in zip((class_one, class_two),('^', 's', 'o'),('blue', 'red', 'green')):
    
plt.scatter(x = X_lda[0:20,[0]], y = X_lda[0:20,[1]], color = 'red', label = 'NSCLC')
plt.scatter(x = X_lda[20:40,[0]], y = X_lda[0:20,[1]], color = 'BLUE', label = 'SCLC')   
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
ax = plt.subplot(121)
plt.title("LDA plot")
plt.scatter(x = X_lda[0:20,[0]], y = np.zeros(20), color = 'red', label = 'NSCLC')
plt.scatter(x = X_lda[20:40,[0]], y = np.zeros(20), color = 'BLUE', label = 'SCLC')
plt.legend(loc= 2)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.grid()
plt.show()

    
    
    
    
    
    
    
    
    
    
