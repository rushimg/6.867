from numpy import *
from plotBoundary import *
from SVM_MULTI import SVM
import numpy as np

# parameters
name = 'kaggle2'
print '======Training======'
# load data from csv files
train = loadtxt('data/'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 1:55].copy()
Y = train[:, 55:56].copy()

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])
alphas = [None] * 7
w_0s = [None] * 7
y_trains = [None] * 7 #list()
# Carry out training, primal and/or dual
for i in range(1,7):
	#print Y
	Y_alt = np.matrix([float(1) if y == i else float(-1) for y in Y]).T
	#print Y_alt
	y_trains[i] = Y_alt 
	C = 1
	svm = SVM(X,Y_alt,C)
	alphas[i], w_0s[i] = svm.train()

# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
	val = [0] * 7
	for i in range(1,7):
		val[i] =svm.test(x, alphas[i], w_0s[i], y_trains[i], X)
	#print np.array(val).argsort()[::-1] 
	return np.array(val).argsort()[::-1][0]
	#return svm.test_gold(x,model)


# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')



print '======Validation======'
# load data from csv files
validate = loadtxt('data/'+name+'_test.csv')
X = validate[:, 1:55]
Y = validate[:, 55:56]

#X = np.array([[1.0,2.0],[2.0,2.0],[0.0,0.0],[-2.0,3.0]])
#Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])

# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

