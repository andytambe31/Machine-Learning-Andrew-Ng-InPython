# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 18:43:04 2021

@author: user
"""

# ---------------------------------------------------------------------------------------------------------
"""
This file contains the code to help you get started with the linear exercise

Here, 

X refers to the population size in 10,000s
Y refers to the profit in $10,000s

"""
# ---------------------------------------------------------------------------------------------------------

# Initializations
import pandas as pd
import numpy as np
from computeCost import *
from gradientDescent import *
from matplotlib import pyplot as plt

'''  ==================== Part 1: Basic Function ==================== '''

# Importing data from txt file
#data = pd.read_csv('ex1data1.txt')
data = pd.read_csv('ex1data1.csv')
#data.head()

# Spliting the data into X and y
X = pd.DataFrame(data.iloc[:,0])
y = pd.DataFrame(data.iloc[:,1])

'''X = pd.DataFrame(data.loc[:,'pop'])
y = pd.DataFrame(data.loc[:,'prof'])'''

# Number of training examples
m = len(y)
'''======================= Part 2: Plotting ======================= '''

# Plot the data
#plt.scatter(X,y) #Alternate syntax: data.plot.scatter(x='pop',y='prof')

''' =================== Part 3: Cost and Gradient descent =================== '''

#Adding X_0 as 1 for the bias
X.insert(0,'x_0',pd.DataFrame({'x_0':[1 for i in range(m)]}),True)

# Coeffiencts for the model
theta = pd.DataFrame([[0],[0]])

# Set the number of iterations for gradient descent
iterations = 1500
alpha = 0.01


print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J.iloc[0]);
print('Expected cost value (approx) 32.07\n');


# further testing of the cost function
theta = pd.DataFrame({'par':[-1,2]})
J = computeCost(X, y, theta);
print('\nWith theta = [-1 ; 2]\nCost computed = ', J.iloc[0]);
print('Expected cost value (approx) 54.24\n');


print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
print('Theta found by gradient descent:\n');
print(theta.iloc[0,0],'\n',theta.iloc[1,0],'\n');
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n\n');

# ------------- Print the predicted values --------------
predict = np.dot(X,theta)
plt.plot(X.iloc[:,1],predict)
plt.scatter(X.iloc[:,1],y,color = 'hotpink')

''' =================== Part 4: Predictions ==================='''

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(pd.DataFrame([[1, 3.5]]),theta);
print('For population = 35,000, we predict a profit of',predict1[0,0]*10000);
predict2 = np.dot(pd.DataFrame([[1, 7]]),theta);
print('For population = 70,000, we predict a profit of',predict2[0,0]*10000);










