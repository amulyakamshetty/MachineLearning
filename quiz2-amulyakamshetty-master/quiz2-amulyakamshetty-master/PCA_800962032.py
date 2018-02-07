# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:52:45 2017

@author: Amulya
"""

import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
from matplotlib.mlab import PCA
import matplotlib.pyplot as   plt

csv = genfromtxt('C:\Users\Amulya\Downloads\dataset_1.csv', delimiter=",")
# print(csv)
# csv[1:1001,0:2]#1:1000 to start from index 1 without the labels(1000 entries)
# print(csv)

x = csv[1:1001, 0]
print("-----------X--------------")
print(x)

y = csv[1:1001, 1]
print("-----------Y--------------")
print(y)

z = csv[1:1001, 2]
print("-----------z--------------")
print(z)

np.mean(x)
mean_centre_x = x - np.mean(x)
# print(mean_centre_x)#printing

mean_squares = np.square(mean_centre_x)

sum_mean_squares = np.sum(mean_squares)

# print(sum_mean_squares/1000)

np.mean(y)
mean_centre_y = y - np.mean(y)
# print(mean_centre_y)#printing

np.mean(z)
mean_centre_z = z - np.mean(z)
# print(mean_centre_z)#printing

A = np.array([mean_centre_x, mean_centre_y, mean_centre_z])
AT = np.transpose(A)
print("A=")
print(A)
print("AT=")
print(AT)
print("A x AT=")
print(A.dot(AT) / 1000)
covA = A.dot(AT) / 1000

print("EIGEN VALUES")
print(np.linalg.eig(covA))
EIG = np.linalg.eig(covA)
print("eigen values")
print(EIG[1])
EIG1T = np.transpose(EIG[1])
print(EIG1T)
print(EIG[0].argsort()[-2:][::-1])
PC = []
for j in (EIG[0].argsort()[-2:][::-1]):
    PC.append(np.transpose(EIG1T[j]))
PC = np.array(PC)
print("----------------------PC---------------------------")
print(np.transpose(PC))

print(AT.dot(np.transpose(PC)))
print(PC[0])
print(PC[1])

fig = plt.figure()
aX = fig.add_subplot(111)
aX.scatter(np.transpose(AT.dot(np.transpose(PC)))[0], np.transpose(AT.dot(np.transpose(PC)))[1])
fig.show()

print(np.var(mean_centre_x))
print("variance of x")
print(np.var(x))

print("variance of y")
print(np.var(y))

print("variance of z")
print(np.var(z))

print("covariance of x and x")
print(np.cov(mean_centre_x, mean_centre_x))

print("COVARIANCE BETWEEN X AND Y")
print(np.cov(x, y))

print("COVARIANCE BETWEEN Y AND Z")
print(np.cov(y, z))

q = np.array([[0, -1], [2, 3]])
EIG1 = np.linalg.eig(q)
print("---------------")
print("QUESTION 3 : Finding eigen values :")
print(EIG1)
print("----same as manual solution of eigen values------")
