# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 01:07:13 2017

@author: Amulya
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import datasets




#loading the dataset
def get_dataset(dataset):
    iris = datasets.load_iris()
    X = iris.data
    #feature_names = iris.feature_names
    #y = iris.target
    #target_names = iris.target_names
    #data = genfromtxt('/Users/Amulya/Downloads/SCLC_study_output_filtered.csv', delimiter=',')
    #print(data)
    #data.shape
    dataset = X

    #cleandata = data[1:41,1:52]
    #print(cleandata)

    #dataset = cleandata
    print(dataset)
    return dataset
    #return np.loadtxt(dataset)

#calculate euclidian distance
def euclid_dist(a, b):
    return np.linalg.norm(a-b)


def plot(dataset, all_centroids, clust):
    colors = ['r', 'g','y']

    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        near_instance = [i for i in range(len(clust)) if clust[i] == index]
        for current_index in near_instance:
            ax.plot(dataset[current_index][0], dataset[current_index][1], (colors[index] + 'o'))

    set_of_points = []
    for index, centroids in enumerate(all_centroids):
        for inside, point in enumerate(centroids):
            if index == 0:
                set_of_points.append(ax.plot(point[0], point[1], 'bo')[0])
            else:
                set_of_points[inside].set_data(point[0], point[1])
                print("centroids {} {}".format(index, point))

                plt.pause(0.8)


def kmeans(k, e=0, d='euclid_dist'):
    all_centroids = []
    if d == 'euclid_dist':
        dist_method = euclid_dist
    #dataset = get_dataset('C:\Users\Amulya\Downloads\SCLC_study_output_filtered_2.csv')
    iris = datasets.load_iris()
    X = iris.data
    dataset = X
    num_of_inst, number_of_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_of_inst - 1, size=k)]
    all_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    clust = np.zeros((num_of_inst, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > e:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)

            clust[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, number_of_features))

        for index in range(len(prototypes)):
            near_instance = [i for i in range(len(clust)) if clust[i] == index]
            prototype = np.mean(dataset[near_instance], axis=0)
            
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        all_centroids.append(tmp_prototypes)

    # plot(dataset, all_centroids, clust)

    return prototypes, all_centroids, clust


def execute():
    #dataset = get_dataset('C:\Users\Amulya\Downloads\SCLC_study_output_filtered_2.csv')
    iris = datasets.load_iris()
    X = iris.data
    dataset = X
    centroids, all_centroids, clust = kmeans(3)
    plot(dataset, all_centroids, clust)

execute()