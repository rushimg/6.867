import pdb
import random
import pylab as pl
import numpy as np
from scipy.optimize import fmin_bfgs
M = 2
numIterations= 100000
alpha = 0.00001
# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    #print phi
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    #print 'w', w
    #print 'X', X
    # produce a plot of the values of the function 
    pts =np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    #print 'pts', pts
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    #EvaluateModel(X,w,order)
    #t_eval = EvaluateModel(, w, order) 
    theta = np.ones(X.shape)
    print gradientDescent(X, Y, theta, sse_deriv,alpha, numIterations)
    #print 'sse', sse(t_test, Yp)	
    pl.plot(pts, Yp.tolist()[0])
    pl.show()

def gradientDescent(x, y, theta, derivative,alpha, numIterations):
    print x.shape
    print theta.shape
    m,n = x.shape
    xTrans = x.transpose()
    for i in range(0, numIterations):
        #print theta
        hypothesis = np.dot(x.transpose(), theta)
        print hypothesis
	#print hypothesis.shape
	#print y.shape
	loss = sse(hypothesis, y)
	#loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = loss
	#cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = derivative(xTrans,loss)
	#gradient = np.dot(xTrans, loss) / m
       # update
        theta = theta - alpha * gradient /m
    return theta


def EvaluateModel(x, w, order):
    result = np.zeros((len(x)))
    for i in range(len(x)):
        for j in range(order):
            if j==0:
                result[i] = w[j,0] # the first basis fn is allways constant
            else:
                result[i] += w[j,0]*x[i]**j
    return result

#sum of squares error
def sse(x1,x2):
	#print x1-x2
	return ((x1-x2)**2).sum()

def sse_deriv(x1,x2):
	return (2*(x1-x2)).sum()

def designMatrix(x, order):
	try:
		m = x.shape[0]
	except:
		m = len(x)
	Phi = np.mat(np.zeros((m, order)))
	for i in range(m):
		 for j in range(order):
			if j==0:
				Phi[i,j] = 0
			else:
				Phi[i,j] = x[i]**j	
	return Phi
	
def regressionFit(X, Y, phi):
	psuedo_inv = np.linalg.pinv(phi)
	#t = np.mat(Y).T
	return psuedo_inv*Y

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('homework1/curvefitting.txt')

def regressAData():
    return getData('homework1/regressA_train.txt')

def regressBData():
    return getData('homework1/regressB_train.txt')

def validateData():
    return getData('homework1/regress_validate.txt')

if __name__ == "__main__":
	X,Y = bishopCurveData()
	regressionPlot(X, Y, M)
