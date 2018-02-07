# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:29:05 2017

@author: Amulya
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from numpy import genfromtxt
import math

X = np.array([[1],[0.05],[0.1]])
#print X
y = np.array([[0.01],[0.99]])
#print y

model = {}



#initialize theta_ome and theta_two to be between 0 and 1
theta_one = np.random.random((2,3))  
theta_two = np.random.random((2,3))

def sigmoid(theta, x): 
    a =( 1/(1 + (np.exp(-((theta).dot(x))))))
    return a

def sigmoid_der(z):
    b = (z * (1 - z))
    return b
    
j_theta_array = []
z_array = []
theta_one_array = []
theta_two_array = []
theta_one_00 = []
theta_one_01 = []
theta_one_02 = []
theta_one_10 = []
theta_one_11 = []
theta_one_12 = []

theta_two_00 = []
theta_two_01 = []
theta_two_02 = []
theta_two_10 = []
theta_two_11 = []
theta_two_12 = []


for z in range (0,1000) :
 print "ITERATION : " , z   
 #print "--------------theta one --------------------------------"
 #print theta_one
 #print "--------------theta two --------------------------------"
 #print theta_two





#implement FP using sigmoid function to get hidden layer units
 a_two = sigmoid(theta_one, X)
#print a_two                

#a_two =( 1/(1 + (np.exp(-((theta_one).dot(X))))) )
#print "--------------a for hidden layer --------------------------------"
#print a_two
#print np.shape(a_two)

#a_two_one = a_two[0]
#print a_two_one
#a_two_two = a_two[1]
#print a_two_two

 a_two_new = np.append([1],[a_two])
#print a_two_bias
 a_two = a_two_new.reshape(3,1)
 #print "--------------a for hidden layer with bias ----------------------------"
 #print a_two
#print np.shape(a_two_bias)


#compute output units using FP sigmoid function on hidden layer units
 a_three = sigmoid(theta_two, a_two)
#print a_three


#a_three = ( 1/(1 + (np.exp(-((theta_two).dot(a_two_bias))))) )
 print "-------------- a for output layer --------------------------------"
 print a_three

 y_0 = a_three[0]
 y_1 = a_three[1]
#print y_0, y_1

 error = y - a_three
 #print error

#computing cost function
 sum = np.sum([(math.pow((y[0]-y_0),2)),(math.pow((y[1]-y_1),2))])
 j_theta = 0.5 * sum
 print "-------------- cost function J(theta) --------------------------------"
 print j_theta

 j_theta_array.append(j_theta)
 z_array.append(z)
#implement BP to compute partial derivatives
 

#delta_two = (error) * sigmoid_der(a_three)
#print delta_two

#delta_three = (error)*(a_three * (1 - a_three))
 #print "-------------- delta three --------------------------------"
 delta_three_error = a_three - y
 #print delta_three_error
#print delta_two

 delta_three = (delta_three_error) * sigmoid_der(a_three)
 #print delta_three

 delta_two_a = np.dot(theta_two.T, delta_three_error)  
 #print delta_two_a

 delta_two_b = sigmoid_der(a_two)
 #print delta_two_b

#delta_one_b = np.cross((a_two_bias),(([1],[1],[1]) - a_two_bias), axis=0)
#delta_one_b

 delta_two = np.cross((delta_two_a),(delta_two_b),axis=0)
 #print "-------------- delta two --------------------------------"
 #print delta_two

#update theta values
 theta_one = theta_one -  np.dot(X.T, delta_two)
 #print "-------------- updated theta one --------------------------------"
# print theta_one
 theta_one_array.append(theta_one)

 theta_two = theta_two -  np.dot(delta_three, (a_two).T)
 #print "-------------- updated theta two --------------------------------"
 #print theta_two
 theta_two_array.append(theta_two)
#print j_theta_array
 s = theta_one_array[z][0][0]
 theta_one_00.append(s)
 
 t = theta_one_array[z][0][1]
 theta_one_01.append(t)
 
 u = theta_one_array[z][0][2]
 theta_one_02.append(u)
 
 v = theta_one_array[z][1][0]
 theta_one_10.append(v)
 
 w = theta_one_array[z][1][1]
 theta_one_11.append(w)
 
 x = theta_one_array[z][1][2]
 theta_one_12.append(x)
 
 
 
 f = theta_two_array[z][0][0]
 theta_two_00.append(f)
 
 g = theta_two_array[z][0][1]
 theta_two_01.append(g)
 
 h = theta_two_array[z][0][2]
 theta_two_02.append(h)
 
 j = theta_two_array[z][1][0]
 theta_two_10.append(j)
 
 k = theta_two_array[z][1][1]
 theta_two_11.append(k)
 
 l = theta_two_array[z][1][2]
 theta_two_12.append(l)
 



fig = plt.figure(1)
#plt.figure(figsize=(1,1))
ax = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
ax.set_title('cost vs iterations')
#ax.scatter(x,y,color = 'r')
#ax.scatter(x,y_theoretical,color = 'b')
#ax.scatter(pc1,pc2,color = 'black')
plt.xlabel('iterations')
plt.ylabel('cost')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 0.36)
plt.xlim(0,200)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(j_theta_array), max(j_theta_array)+0.0002, 0.1))
ax.plot(z_array, j_theta_array, color='red', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()

fig = plt.figure(2)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 00')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_00), max(theta_one_00)+0.5, 0.5))
bx.plot(z_array,(theta_one_00), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(3)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 01')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_01), max(theta_one_01)+0.5, 0.5))
bx.plot(z_array,(theta_one_01), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()

fig = plt.figure(4)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 02')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_02), max(theta_one_02)+0.5, 0.5))
bx.plot(z_array,(theta_one_02), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()

fig = plt.figure(5)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 10')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_10), max(theta_one_10)+0.5, 0.5))
bx.plot(z_array,(theta_one_10), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()

fig = plt.figure(6)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 11')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_11), max(theta_one_11)+0.5, 0.5))
bx.plot(z_array,(theta_one_11), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()

fig = plt.figure(7)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta one 12')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_one_12), max(theta_one_12)+0.5, 0.5))
bx.plot(z_array,(theta_one_12), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(8)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 00')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_00), max(theta_two_00)+0.5, 0.5))
bx.plot(z_array,(theta_two_00), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(9)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 01')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_01), max(theta_two_01)+0.5, 0.5))
bx.plot(z_array,(theta_two_01), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(10)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 02')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_02), max(theta_two_02)+0.5, 0.5))
bx.plot(z_array,(theta_two_02), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(11)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 10')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_10), max(theta_two_10)+0.5, 0.5))
bx.plot(z_array,(theta_two_10), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(12)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 11')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_11), max(theta_two_11)+0.5, 0.5))
bx.plot(z_array,(theta_two_11), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()


fig = plt.figure(13)
bx = fig.add_subplot(2,1,1)
#ax = fig.add_axes([0.5, 0.5, 0.5, 0.5])
#ax.set_title('theta one vs iterations')
plt.xlabel('iterations')
plt.ylabel('theta two 12')
#fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,4)
#for z, j_theta in j_theta_array:
 #    print "This is the cost function value in " + str(z) + ' iteration : ', j_theta
plt.axis('tight')
plt.ylim(0, 2)
plt.xlim(0,900)
plt.xticks(np.arange(min(z_array), max(z_array)+1, 50.0))
plt.yticks(np.arange(min(theta_two_12), max(theta_two_12)+0.5, 0.5))
bx.plot(z_array,(theta_two_12), color='green', linewidth=1)
#ax.set_aspect('equal', 'box')
fig.show()







