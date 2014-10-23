import pdb
import random
import pylab as pl
import numpy as np
from scipy.optimize import fmin_bfgs
import math
M = 4
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
    w = w/1.7
    #w = w*x.shape[1]
    #print w
    #print 'w', w
    #print 'X', X
    # produce a plot of the values of the function 
    pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    #print 'pts', pts
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    #EvaluateModel(X,w,order)
    #t_eval = EvaluateModel(, w, order) 
    #alpha = 0.001
    #convergence = .0001
    #print gradientDescent(X ,Y , None, None, alpha, convergence)
    #print gradientDescent(X, Y, sse_deriv,alpha, numIterations)
    #print 'sse', sse(t_test, Yp)	
    pl.plot(pts, Yp.tolist()[0])
    pl.show()

def l2_dist(x1,x2):
        dist = sum((x1-x2)**2)
        #dist = 0
        #m = np.shape(x1)[0]
        #for i in range(0,m):
        #       dist += (x1[i] - x2[i])*(x1[i] - x2[i])
        return math.sqrt(dist)

def gradientDescent(x ,y , initial_guess, objective_func, step_size, convergence):
	#def gradientDescent(x, y, derivative,alpha, numIterations):
	count_iter = 0 # keep track of number of iterations
        #x = np.array(initial_guess)
        print x.shape
        xTrans = x.transpose()
        m,n = x.shape
        theta = np.ones(n)
        print theta.shape
        error = 1
        while abs(error) > convergence:
                print theta
                count_iter+=1
                h0 = np.dot(x.T, theta) # h0
                #y = func(x)
                #loss = h0 - y
                theta_new = theta - step_size * sse_deriv(xTrans,h0-y)/m
                error = l2_dist(theta_new,theta)
                theta = theta_new
        return theta,count_iter	
	'''	    
    m,n = x.shape
    theta = np.ones(n)
    xTrans = x.transpose()
    for i in range(0, numIterations):
        #print theta
        hypothesis = np.dot(x, theta)
        #print hypothesis
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
	'''

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
	return (-2*(x1-x2)).sum()

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
