# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:02:02 2017

@author: Amulya
"""

import math
import numpy as np
from numpy import genfromtxt


def get_dataset(dataset):
    #return np.loadtxt(dataset)
    data = genfromtxt('/Users/Amulya/Downloads/SCLC_study_output_filtered.csv', delimiter=',')

    #data=get_dataset('C:\Users\Amulya\Downloads\SCLC_study_output_filtered_2.csv')
    cleandata = data[1:41,1:52]
    #print(cleandata)

    dataset = cleandata

    return dataset

def dist(a,b):
    a=float(a[0])-float(b[0])
    a=a*a
    b=float(a[1])-float(b[1])
    b=b*b
    dist=round(math.sqrt(a+b),2)
    return dist

def mini(matrix):
    p=[0,0]
    mn=1000
    for i in range(0,len(matrix)):        
        for j in range(0,len(matrix[i])):            
            if (matrix[i][j]>0 and matrix[i][j]<mn):
                mn=matrix[i][j]
                p[0]=i
                p[1]=j
    return p 

def load_data(name):
    return np.loadtxt(name)
            
def newpoint(pt):
    a=float(pt[0][0])+float(pt[1][0])
    a=a/2
    b=float(pt[0][1])+float(pt[1][1])
    b=b/2
    midpoint=[a,b]
    return midpoint

if __name__ == '__main__':    
    #n=int(raw_input("Enter number of points.: "))
    data=load_data('/Users/Amulya/Downloads/SCLC_study_output_filtered.csv')
    data.shape

    cleandata = data[1:41,1:52]
    #print(cleandata)

    dataset = cleandata
    
    data=data[0:5,]
    points= list() 
    points=data.tolist()
    type(points)
    outline='['
    
    names={}
    
    for i in range(0,len(points)):
        names[str(points[i])]=(i)
    l=0
    while(len(points)>1):
        l=l+1
        matrix=list()
        print('Dist matrix no. ' + str(l) + ' :')
        for i in range(0,len(points)):
            m=[]
            for j in range(0,len(points)):
                m.append(0)
            for j in range(0,len(points)):
                m[j]=dist(points[i],points[j])
            print(str(m))
            matrix.append(m)
            
            
            
        
        m=mini(matrix)
        pts=list()
        pa=points[m[0]]
        pts.append(pa)
        points.remove(pa)
        pb=points[m[1]-1]
        pts.append(pb)
        points.remove(pb)   
        mp=newpoint(pts)
        points.append(mp)    
        ca=names.pop(str(pa))
        cb=names.pop(str(pb))
        names[str(mp)]="["+str(ca)+str(cb)+"]"    
        outline=names[str(mp)]
        
    print("Cluster is :",names[str(mp)])