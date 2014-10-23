import pdb
import random
import pylab as pl
import numpy as np
from scipy.optimize import fmin_bfgs
M = 2
LAMBDA = .00001
# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    #print X.shape
    #print Y.shape
    #pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    phi = designMatrix(X, order)
    w  =  regressionFit(X, Y, phi)
    w_o, w_ridge = ridgeRegression(X, Y,phi, LAMBDA)
    
    pts =np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    
    #Yp = pl.dot(w_ridge.T, designMatrix(pts, order).T)
    #print w_o.shape
    #print w_ridge.shape
    #print X.shape
    #print Y.shape
    #Y_calc = w_ridge*X.T
    #print Y_calc.shape
    #Y_orig = pl.dot(w.T, designMatrix(pts, order).T)
    #print ridge_error(X,Y,w_ridge,LAMBDA)
    #Y_calc = w_o + w_ridge.T*X
    print sse(w_ridge,w)
    #print w_o.shape
    #print w_ridge.shape
    #print X.shape
    #print Y.shape
    #pl.plot(pts, Yp.tolist()[0])
    #pl.show()

def ridge_error(X,Y,W_ridge,lam):
	Z = X - X.sum()/(X.shape[0] * X.shape[1])
        Y_c = Y - Y.sum()/(Y.shape[0] * Y.shape[1])
        Eridge = (Y_c - Z*W_ridge).T * (Y_c - Z*W_ridge)  + lam*W_ridge.T*W_ridge
	return Eridge

def sse(x1,x2):
	#print x1-x2
	return ((x1-x2)*(x1-x2).T).sum()

def sse_deriv(x1,x2):
	return (2*(x1-x2)).sum()

def designMatrix(x, order):
	try:
		m,n = x.shape
	except:
		m = len(x)
	Phi = np.mat(np.zeros((m, order)))
	#Phi =np.empty((m,n,order))
	for i in range(m):
		 for j in range(order):
			if j==0:
				Phi[i,j] = 0
			else:
				Phi[i,j] = sum(x[i])**j	
	return Phi
	
def regressionFit(X, Y, phi):
        psuedo_inv = np.linalg.pinv(phi)
        #t = np.mat(Y).T
        print psuedo_inv.shape
	print Y.shape
	return psuedo_inv*Y.T

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

def getBlogData():
	X_train = np.genfromtxt('blogData/x_train.csv', dtype=float, delimiter=',')
	Y_train = np.genfromtxt('blogData/y_train.csv', dtype=float, delimiter=',')
	X_valid = np.genfromtxt('blogData/x_val.csv', dtype=float, delimiter=',')
	Y_valid = np.genfromtxt('blogData/y_val.csv', dtype=float, delimiter=',')
	X_test = np.genfromtxt('blogData/x_test.csv', dtype=float, delimiter=',')
	Y_test = np.genfromtxt('blogData/y_test.csv', dtype=float, delimiter=',')
	return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

if __name__ == "__main__":
	X_train,Y_train,X_valid,Y_valid,X_test,Y_test = getBlogData()	 
	#X,Y = bishopCurveData()
	#X, Y = regressAData()
	#regressionPlot(X, Y, M)
	#X, Y = regressBData()
        #regressionPlot(X, Y, M)
	#X, Y = validateData()
        regressionPlot(X_train, Y_train, M)
