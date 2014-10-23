import pdb
import random
import pylab as pl
import numpy as np
from scipy.optimize import fmin_bfgs
M = 6
LAMBDA = 1
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
    #w = ridgeRegression(X, Y, 1)
    #print w.shape
    w  =  regressionFit(X, Y, phi)
    w_o, w_ridge = ridgeRegression(X, Y,phi, LAMBDA)
    #Yp = w_o + sum(w_ridge*X)
    #print Yp.shape
    #print w.shape
    #print w_ridge.shape
    
    #print 'ridge_error', sum(sum(ridge_error(X,Y,w,LAMBDA)))
    #print w_o.shape
    #print y_pred.shape
    #print w_1.shape
    #pts =np.array([[p] for p in pl.linspace(min(X), max(X), 10)])
    pts =np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    #pts_10 = np.array([[p] for p in pl.linspace(min(X), max(X), 10)])
    #print w_ridge.shape
    #y_pred = w_o[0] + sum(w_ridge[0]*pts_10)
    #print y_pred
    #print Y
    #print y_pred.shape
    #pl.plot(pts_10, y_pred.tolist())
    #print 'pts', pts
    #Yp = w_o + sum(w_ridge*X)
    #print pts.shape 
    #print Yp.shape
    Yp = pl.dot(w_ridge.T, designMatrix(pts, order).T)
    #print Yp.shape
    #EvaluateModel(X,w,order)
    #t_eval = EvaluateModel(, w, order) 
    #theta = np.ones(X.shape)
    #print gradientDescent(X, Y, theta, sse_deriv,alpha, numIterations)
    #print 'sse', sse(t_test, Yp)	
    pl.plot(pts, Yp.tolist()[0])
    pl.show()
    #print Y.shape
    #print Yp.shape
    #print sse(Y, Yp.transpose())
#w.transpose()*X + w_1)
#sum of squares error
def ridge_error(x,y,W_ridge,lam):
	Z = X - X.sum()/(X.shape[0] * X.shape[1])
        Y_c = Y - Y.sum()/(Y.shape[0] * Y.shape[1])
        Eridge = (Y_c - Z*W_ridge).T * (Y_c - Z*W_ridge)  + lam*W_ridge.T*W_ridge
	return Eridge

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

def ridgeRegression(X, Y, phi,lam):
	Z = phi - phi.sum()/(phi.shape[0] * phi.shape[1])
	Y_c = Y - Y.sum()/(Y.shape[0] * Y.shape[1])
	#print ((Z.transpose())*Z).shape
	#print lam*np.identity(Z.shape[0]).shape
	#W_ridge = np.linalg.inv(Z.transpose()*Z+lam*np.identity(Z.shape[0]))*Z.transpose()*Y_c
	#print 'shape', W_ridge
	#w = W_ridge
	#Eridge = (Y_c - Z*W_ridge).T * (Y_c - Z*W_ridge)  + lam*W_ridge.T*W_ridge
	#return (Y.sum()/(Y.shape[0] * Y.shape[1]))-W_ridge.transpose()*(X.sum()/(X.shape[0] * X.shape[1]))	
	#W_o = (Y.sum()/(Y.shape[0] * Y.shape[1]))-w.transpose()*(X.sum()/(X.shape[0] * X.shape[1]))
	#Eridge = (Y_c - Z*W_o).T * (Y_c - Z*W_o)  + lam*W_o.T*W_o
	W_ridge = np.linalg.inv(Z.T*Z-(lam*np.identity(Z.shape[1])))*phi.T*Y_c
	w = W_ridge
	#w_ridge = (phi.T*phi).inverse()*Y
	W_o = (Y.sum()/(Y.shape[0] * Y.shape[1]))-w.transpose()*(X.sum()/(X.shape[0] * X.shape[1]))
	return W_o, W_ridge

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
