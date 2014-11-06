from numpy import *
from plotBoundary import *
import numpy as np
# import your LR training code
from LR import LR
# parameters
name= 'stdev2'
print '======Training======'
# load data from csv files
train = loadtxt('newData-2/data_'+name+'_train.csv')
#train = loadtxt('data/data_'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])

# Carry out training.
#L = .00000001
L = 0 
lr = LR(X,Y,L)
model = lr.train_gold()
'''
[[ 0.89444823  0.19756899]]
[-0.24464889]
model = lr.train_gold()
print model.coef_
print model.intercept_
'''
# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
        return lr.test_gold(x, model)

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')
'''
print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
'''
