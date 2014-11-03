import numpy as np
from cvxopt import matrix, solvers
import cvxopt
import random

class SVM:
	def __init__(self, X, Y, C):
		self.X_train = X
		self.Y_train = Y 
		self.C = C
		#print solvers.options
		#self.kernel = kernel

	#linear kernel
	def kernel(self,X,X_prime):
		return np.dot(X,X_prime)

	def train_gold(self):
		''' sklearn imlplementation to check results against '''
		from sklearn import svm
		y = self.Y_train
                X = self.X_train
		model = svm.LinearSVC(C=self.C)
		return model.fit(X,y)
	
	def test_gold(self,x_test,model):
		return model.predict(x_test)

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
        	solvers.options['feastol'] = 1e-8
		solution = solvers.qp(P, q, G, h, A, b)
		self.alphas = np.array(solution['x'])
		#print 'alphas',alphas		
		#print 'y', y
		# calculate w
		#self.w = (alphas*(y*X))
		#print self.w.shape
		#print y.shape
		#print alphas.shape
		
		#self.w = np.outer(y,alphas)
		#print 'w', self.w
		# calculate w_0
		#self.w_0 = 1 	
		return True
	
	def test(self, x_test):
		#print 'test'
		#print self.w
		#print (np.dot(self.Y_train*self.alphas.T))
		#print self.alphas.shape
		#print self.X_train.shape
		#print x_test
		#w = self.Y_train*self.a	
		#val = self.Y_train*self.alphas*self.X_train.T*x_test
		val = 0
		for i in range(0,len(self.alphas)):
			#print i	
			val += self.alphas[i] * self.Y_train[i] * (np.dot(x_test.T, self.X_train[i]))
		
		w_0 = 0
		for j in range(0,len(self.alphas)):	
			inner_sum = 0
			for i in range(0,len(self.alphas)):
				inner_sum += self.alphas[i] * self.Y_train[i] * (np.dot(self.X_train[j].T, self.X_train[i]))
			w_0 += (self.Y_train[j]-inner_sum)
			#w_0 = self.Y_train[i] - val
		w_0 = w_0/len(self.alphas)

		val += w_0
		
		#print np.dot(self.alphas,self.Y_train.T)
		#print np.dot(self.X_train,x_test.T)
		#val = np.dot((self.alphas*self.Y_train),(self.X_train.T,x_test))
		#val = self.w * (np.dot(x_test,x_test.T))
		#val = random.uniform(-1, 1)
		#print 'val',val
		if val > 0:
			return 1
		elif val< 0:
			return -1
		else:
			return 0

	def error(self):
		return True	
