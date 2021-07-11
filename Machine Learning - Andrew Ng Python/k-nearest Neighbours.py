# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 03:58:44 2021

@author: user
"""


# Code for finding the k nearest neighbours

# -----------------------------------------------

# Below code works on the cordinate data of 2 points

# -----------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import seaborn 

# Import or declare data
rand = np.random.RandomState(42)
X = np.random.randn(10,2)

# Plot data
plt.scatter(X[:,0],X[:,1],s=100)

# Find the difference in points
differences = X[:,np.newaxis,:] - X[np.newaxis,:,:]

# Square the differences
sq_diff = differences**2

# Sum the coordinate differences to get the squared distance
dist_sq = sq_diff.sum(-1)

# Sort the array according to distance to find nearest neighbours
nearest = np.argsort(dist_sq, axis=1)

# Selecting the k nearest
K = 2
nearest_partition = np.argpartition(nearest,K+1,axis=1)

# Plotting the K-nearest neighbours

for i in range(X.shape[0]):
    for j in nearest_partition[i,:K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')