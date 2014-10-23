import numpy as np
from cvxopt import matrix, solvers
import cvxopt

class SVM:
	def __init__(self, X, Y, C):
		self.X_train = X
		self.Y_train = Y 
		self.C = C
		#self.kernel = kernel

	#linear kernel
	def kernel(self,X,X_prime):
		return np.dot(X,X_prime)

	def train(self):
		y = self.Y_train 
		X = self.X_train
		n_samples = self.Y_train.shape[0]
		
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
    			for j in range(n_samples):
	        		K[i,j] = self.kernel(X[i], X[j])

		P = cvxopt.matrix(np.outer(y,y) * K)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1,n_samples))
		b = cvxopt.matrix(0.0)
		G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
		h = cvxopt.matrix(np.zeros(n_samples))		
        	solution = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(solution['x'])
		return alphas
	
	def test(self, X_test, Y_test):
		return Y_test


	def error(self):
		return True	
