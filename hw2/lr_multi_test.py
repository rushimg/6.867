from numpy import *
from plotBoundary import *
import numpy as np
# import your LR training code
from LR_MULTI import LR
from sklearn.preprocessing import OneHotEncoder

# parameters
name= 'kaggle'
print '======Training======'
train = loadtxt('data/'+name+'_train.csv')
# load data from csv files
#train = loadtxt('newData-2/data_'+name+'_train.csv')
#train = loadtxt('data/data_'+name+'_train.csv')
X = train[:,1:55]
Y = train[:,55:56]

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])

# Carry out training.
#L = .00000001
enc = OneHotEncoder()
enc.fit(Y)

Y = enc.transform(Y)
print Y.shape
'''
Y_act = np.array([Y.shape[0],7])
count_i = 0
for i in  Y:
	count_j = 1
	for j in range(1,8):
		if count_j == i[0]:
			print count_i
			print count_j
			Y_act[count_i,count_j-1] = 1
			count_j += 1
	count_i += 1
	#temp = np.zeros([7])
	#temp[i[0]] = 1
	#Y_act[count_i] = temp
	#count_i += 1
'''
L = 0 
lr = LR(X,Y,L)
lr.train()
'''
[[ 0.89444823  0.19756899]]
[-0.24464889]
model = lr.train_gold()
print model.coef_
print model.intercept_
'''
# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
	ret = lr.test(x)
	print ret
        return ret

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
