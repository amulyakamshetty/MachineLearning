# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:25:41 2017

@author: Amulya
"""


#Import required libraries 
#import pandas as pd #loading data in table form  
#import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import math


#Reading data 

my_data = np.loadtxt('C:\Users\Amulya\Desktop\ML_codes\iris.csv',skiprows=51,usecols=range(3,5),delimiter=',')
print ("Data present in the .csv file: \n")

#scaling
petal_length = my_data[:,0]
new_pl = (petal_length - min(petal_length))/(max(petal_length) - min (petal_length))
new_pl = new_pl.reshape(100,1)


petal_width = my_data[:,1]
new_pw = (petal_width - min(petal_width))/(max(petal_width) - min (petal_width))
new_pw = new_pw.reshape(100,1)




z = np.ones((100,1))
my_data_new = np.concatenate((z, new_pl, new_pw), axis = 1)
print my_data_new
#print np.shape(my_data)

y1 = np.ones((50,1))
y2 = np.zeros((50,1))
y = np.concatenate((y1, y2), axis = 0)
print "Y( output array ):"
print y
#print np.shape(y)

#initialize theta_ome and theta_two to be between 0 and 1
theta_one = np.random.random((2,3))  
theta_two = np.random.random((1,3))
#print theta_one

def sigmoid(theta, x): 
    a =( 1/(1 + (np.exp(-((theta).dot(x))))))
    return a

def sigmoid_der(z):
    b = (z * (1 - z))
    return b

j_theta = []
z_array = []
theta_one_array = []
theta_two_array = []

error_arr = []

j_theta_tot = 0
global error_sum
error_sum = 0
#error = 0

#AS DONE IN SLIDES : 
    
for index_iter in range (1, 100) :
 DELTA_1 = 0
 DELTA_2 = 0
#TRAINING
 print "ITERATION :", index_iter+1 
 for z in range (0,99) :    
  
#forward prop

  my_data_del = my_data_new
  #my_data_del = np.delete(my_data_new, (z) ,axis=0) 
  #np.shape(my_data_del) 
#implement FP using sigmoid function to get hidden layer units
  a_two = sigmoid(theta_one, my_data_del[z])
 #print a_two
  a_two_new = np.append([1],[a_two])
 #print a_two_new   #hidden layer units with bias
  a_two = a_two_new.reshape(3,1)
 #print "a_two"
 #print a_two

  a_three = sigmoid(theta_two, a_two)
 #print "a_three"
 #print a_three  #a_three = h(theta)
 
  #if a_three > 0.5 :
   #a_three_new = 1
  #else :
   #a_three_new = 0
  #print  "   Expected output :" ,  y[z]
 
  #print  "   Predicted output :" , a_three_new
  
 
  #error = a_three - y[z]
  #print "   Error :" , error
  
  
  #error_sum += error 
 
 #computing cost function
 
 #if a_three_new == 1 :     
  # a = (y[z]*math.log(a_three))
  #b = 0

 #else :
  #a = 0
  #b =  ((1 - y[z])*math.log(1 - a_three) )
  
  #j_theta_i = (a + b) * -1/99
  #print "-------------- cost function J(theta) --------------------------------"
  #print j_theta_i
  #j_theta.append(j_theta_i)
 #j_theta_tot += j_theta_i 
 #print j_theta_tot * -1/99
  #z_array.append(z)
  
  delta_three_error = a_three - y[z]
 #print delta_three_error
 
  delta_three = (delta_three_error) * sigmoid_der(a_three)
 
  delta_two_a = np.dot(theta_two.T, delta_three_error)  
 
  delta_two_b = sigmoid_der(a_two)
 
  delta_two = np.cross((delta_two_a),(delta_two_b),axis=0)
 
 #update theta values
  pp = np.dot(delta_three, (a_two).T)
  DELTA_2 = DELTA_2 + pp
  
  qq = np.dot(my_data_del[z].T, delta_two)
  DELTA_1 = DELTA_1 + qq
  
  
  
  
  #theta_one = theta_one -  np.dot(my_data_del[z].T, delta_two)
# print "-------------- updated theta one --------------------------------"
 #print theta_one
  #theta_one_array.append(theta_one)

  
  #theta_two = theta_two -  np.dot(delta_three, (a_two).T)
 #print "-------------- updated theta two --------------------------------"
 #print theta_two
  #theta_two_array.append(theta_two)
 
  #error_arr.append(error)
 #testing
 d_1 = DELTA_1 / 100
 d_2 = DELTA_2 / 100
  
 theta_one = theta_one - d_1
 theta_two = theta_two - d_2 
 
 a = (y[z]*math.log(a_three))
 b =  ((1 - y[z])*math.log(1 - a_three) )
 j_theta_i = (a + b) * -1/99
 #print "-------------- cost function J(theta) --------------------------------"
 #print j_theta_i
 print a_three
 if a_three > 0.5 :
   a_three_new = 1
 else :
   a_three_new = 0
 print  "   Expected output :" ,  y[z]
 
 print  "   Predicted output :" , a_three_new
  
 if y[z] == a_three_new :
     error = 0
 else :
     error = 1
 print error
 error_sum += error
 
 print "sum of errors : ", error_sum




 
 

 

 
 