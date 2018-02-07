# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 06:25:21 2017

@author: Amulya
"""

import numpy as np # linear algebra
#import math

my_data = np.loadtxt('C:\Users\Amulya\Desktop\ML_codes\iris.csv',skiprows=51,usecols=range(3,5),delimiter=',')
print ("Data present in the .csv file: \n")


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


def sigmoid(theta, x): 
    a =( 1/(1 + (np.exp(-((theta).dot(x))))))
    return a

def sigmoid_der(z):
    b = (z * (1 - z))
    return b
x.tolist()

#calculate H_theta
alpha = 0.2
tita = 0
for k in range (99):
    
    z = sigmoid( tita, np.sum(np.dot(x[k][:,0:2])))
    
    if (x[k][:,2] == 0) :
        tita = np.subtract(tita,(alpha*(z-1)*x[k][:,0:2]).reshape(2,1))
    else :
         tita = np.subtract(tita,(alpha*(z-1)*x[k][:,0:2]).reshape(2,1))
            
        
        