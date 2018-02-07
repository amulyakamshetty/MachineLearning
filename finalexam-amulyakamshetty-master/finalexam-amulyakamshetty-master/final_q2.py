# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 03:15:47 2017

@author: Amulya
"""

import numpy as np # linear algebra
#import math

my_data = np.loadtxt('C:\Users\Amulya\Desktop\ML_codes\iris.csv',skiprows=51,usecols=range(3,5),delimiter=',')
print ("Data present in the .csv file: \n")

#scaling of data
petal_length = my_data[:,0]
new_pl = (petal_length - min(petal_length))/(max(petal_length) - min (petal_length))
new_pl = new_pl.reshape(100,1)


petal_width = my_data[:,1]
new_pw = (petal_width - min(petal_width))/(max(petal_width) - min (petal_width))
new_pw = new_pw.reshape(100,1)




z = np.ones((100,1))
my_data_new = np.concatenate((z, new_pl, new_pw), axis = 1)
print my_data_new
x = my_data_new

y1 = np.ones((50,1))
y2 = np.zeros((50,1))
y = np.concatenate((y1, y2), axis = 0)
print "Y( output array ):"
print y


theta_one = np.random.random((2,3))  
theta_two = np.random.random((1,3))
D_two = 0
D_one = 0


def sigmoid(theta, x): 
    a =( 1/(1 + (np.exp(-((theta).dot(x))))))
    return a

def sigmoid_der(z):
    b = (z * (1 - z))
    return b

def delete_row(dataset, x):
    dataset_new = np.delete(dataset, x ,0)
    return dataset_new

#as done in slides
#TRAINING  
def training() :  
   a_three_arr = []
   a_two_arr = []
   
   theta_one = np.random.random((2,3))  
   theta_two = np.random.random((1,3))   
   for ind in range (1, 100) :
    print "iteration : ", ind
    #x = delete_row(my_data_new, ind)
    tri_1 = 0
    tri_2 = 0
    for i in range(0,99):
        a_one = x[i]
        #p = a_one
        a_two_arr = sigmoid(theta_one, a_one)
        a_two_new = np.append([1],[a_two_arr])
        a_two = a_two_new.reshape(3,1)
        a_three_arr = sigmoid(theta_two, a_two_new)
        
        delta_3 = (a_three_arr - y[i])*( a_two_new)
        
        a = np.dot(theta_two.T, (a_three_arr - y[i]))
        delta_2 = np.cross(a, sigmoid_der(a_two), axis=0)
        
        
        tri_2 =  tri_2 + np.dot(delta_3, a_two_new.T)
        tri_1 = tri_1 + np.dot(([a_one.T]), delta_2)
        print a_three_arr
        D_two =  tri_2/100
        D_one =  tri_1/100

        theta_one = theta_one - 0.1*D_one
       # print theta_one
        theta_two = theta_two - 0.1 *D_two
       # print theta_two



#TESTING
def Testing():
   a_three_arr = []
   a_two_arr = []
   positives = 0
   theta_one = np.random.random((2,3))  
   theta_two = np.random.random((1,3))   
   p_arr = []
    #x = delete_row(my_data_new, ind)
   tri_1 = 0
   tri_2 = 0
   for i in range(0,99):
        a_one = x[i]
        #p = a_one
        a_two_arr = sigmoid(theta_one, a_one)
        a_two_new = np.append([1],[a_two_arr])
        a_two = a_two_new.reshape(3,1)
        a_three_arr = sigmoid(theta_two, a_two_new)
        
        delta_3 = (a_three_arr - y[i])*( a_two_new)
        
        a = np.dot(theta_two.T, (a_three_arr - y[i]))
        delta_2 = np.cross(a, sigmoid_der(a_two), axis=0)
        
        
        tri_2 =  tri_2 + np.dot(delta_3, a_two_new.T)
        tri_1 = tri_1 + np.dot(([a_one.T]), delta_2)
        print "final testing values of testset:"
        print a_three_arr
        if a_three_arr > 0.8:
            p = 1
        else :
            p = 0
        p_arr.append(p)
        
        if p_arr[i]== y[i]:
            positives += 1
        D_two =  tri_2/100
        D_one =  tri_1/100

        theta_one = theta_one - 0.1*D_one
       # print theta_one
        theta_two = theta_two - 0.1 *D_two
       # print theta_two
   print p_arr
   print "NO. of correct values  : ", positives
   #p_a = np.reshape(p_arr, (100,1))
   s = np.sum([p_arr])
   
training()
Testing()
       
        
        