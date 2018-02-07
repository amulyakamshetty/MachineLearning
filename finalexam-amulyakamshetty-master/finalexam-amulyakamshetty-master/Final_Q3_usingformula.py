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
# I tried by calculating the probabilities of getting versicololr and  virginica
my_data = np.loadtxt('C:\Users\Amulya\Desktop\ML_codes\iris.csv',skiprows=51,usecols=range(3,5),delimiter=',')
#my_data = np.loadtxt('C:\Users\Apoorva\Downloads\iris.csv',skiprows=51,usecols=range(3,5),delimiter=',')
print ("Data present in the .csv file: \n")

#scaling
petal_length = my_data[:,0]
new_pl = (petal_length - min(petal_length))/(max(petal_length) - min (petal_length))
new_pl = new_pl.reshape(100,1)


#no scaling of data
#calculated the formula as 
#Prob of being versicolor = e^(44.929- 5.718*petal_length - 10.352*petal_width) / (e^(44.929- 5.718*petal_length - 10.352*petal_width ) + 1)

petal_width = my_data[:,1]
new_pw = (petal_width - min(petal_width))/(max(petal_width) - min (petal_width))
new_pw = new_pw.reshape(100,1)

for x in range(0, len(new_pw)):
    q = 44.929- 5.718*float(petal_length[x]) - 10.352*float(petal_width[x])
    y=math.exp( q )/(1+math.exp(q))
    if (  y > 0.5   ):
        print "+++++"+str(y) + "\tIris-versicolor"
    else:
        print "+++++"+str(y) + "\tIris-virginica"



 
 

 

 
 
