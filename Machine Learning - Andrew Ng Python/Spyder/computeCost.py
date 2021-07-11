# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:31:23 2021

@author: user
"""
import numpy as np

def computeCost(X, y, theta):
    
    # Get the row & column count
    m = len(X)
    
    # Predicted value
    predict = np.dot(X,theta)
    
    J = (predict - y).pow(2).sum(axis = 0).div(2*m)
    
    return J


        

