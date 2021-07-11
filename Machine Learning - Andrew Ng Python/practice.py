# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 01:26:09 2021

@author: user
"""
import numpy as np

# Selection sort
def selectionSort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i],x[swap]) = (x[swap],x[i])
    print(x)

# Bogosort
def bogosort(x):
    
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    print(x)
    

# k-Nearest Neighbour
    
import matplotlib.pyplot as plt
import seaborn # for formatting & styling

rand = np.random.RandomState(42)

X = rand.rand(10,2)
#X = np.array([[5,1,8],[9,8,4],[9,4,1]])

#plt.scatter(X[:,0],X[:,1], s=100)

# for each pair of points, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]

# square the coordinate differences
sq_differences = differences ** 2
sq_differences.shape

# sum the coordinate differences to get the squared distance
dist_sq = sq_differences.sum(-1)
dist_sq.shape

nearest = np.argsort(dist_sq, axis=1)
print(nearest)





