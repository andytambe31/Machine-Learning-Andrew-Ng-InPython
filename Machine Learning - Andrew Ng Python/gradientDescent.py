# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:33:14 2021

@author: user
"""
import pandas as pd
import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, alpha, num_iters):
    
    n = len(theta)
    m = len(X)
    
    # Dataframes
    gradient = pd.DataFrame()
    derivative = pd.DataFrame(theta)
    temp = pd.DataFrame(theta)
    J_history = pd.DataFrame(np.nan, index=range(num_iters), columns=['J_hist'])
    
    #pd.DataFrame(0,index = [i for i range(num_iters)],colummns=['J_his'])
    
    for iter in range(num_iters):
        # Predicted value
        predict = np.dot(X,theta)
        
        
        for j in range(n):
            derivative.iloc[j] = sum((predict - y).iloc[:,0]*(X.iloc[:,j]))/m
            #(predict - y).iloc[:,0]*(X.iloc[:,j]).sum(axis = 0)/m
            
        temp = theta - derivative*alpha
        
        J_history.iloc[iter] = computeCost(X, y, theta)
        
        theta = temp
    
    return theta#[theta,J_history]