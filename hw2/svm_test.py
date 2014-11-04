from numpy import *
from plotBoundary import *
from SVM import SVM
import numpy as np

# parameters
name = 'bigOverlap'
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])

# Carry out training, primal and/or dual
C = 1
svm = SVM(X,Y,C)
svm.train()
#model = svm.train_gold()

# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
	return svm.test(x)
	#return svm.test_gold(x,model)


# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')



print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_test.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])

# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

