import numpy as np
from cvxopt import matrix, solvers
import cvxopt
import random

class SVM:
	'''
	http://cvxopt.org/userguide/coneprog.html#s-qp -> CVX opt documentation
	'''
	def __init__(self, X, Y, C):
		self.X_train = X
		self.Y_train = Y 
		self.C = C
		#print solvers.options
		self.kernel = self.linear_kernel

	#linear kernel
	def linear_kernel(self,X,X_prime):
		return np.dot(X,X_prime)
	
	def rbf_kernel(self,x, y, sigma=5.0):
		#print 'x',x
		#print 'y',y
	    	return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

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
		'''
		http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/
		'''
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
		
		tmp1 = np.diag(np.ones(n_samples) * -1)
                tmp2 = np.identity(n_samples)
                G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
                tmp1 = np.zeros(n_samples)
        	tmp2 = np.ones(n_samples) * self.C
	        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

		#G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1)) # hard margin
		#h = cvxopt.matrix(np.zeros(n_samples)) # hard margin		
       
		#solvers.options['feastol'] = 1e-8
		solvers.options['maxiters'] = 1000
		solution = solvers.qp(P, q, G, h, A, b)
		self.alphas = self.clean_alphas(np.array(solution['x']))
		
		self.calc_w_0()
		return self.alphas, self.w_0
		'''
		num_alphas = 0
		for a in self.alphas:
			if a > 1e-5:
				num_alphas += 1
		print 'num_alphas' ,num_alphas

		self.calc_w_0()
		print 'w_0' , self.w_0
		
		val = np.zeros(2)
                for i in range(0,len(self.alphas)):
                        #print i        
                        val += self.alphas[i] * self.Y_train[i] * self.X_train[i]
		print 'geometric margin', 1/(val[0]*val[0] + val[1]+val[1])
		print 'c', self.C
		'''
		return True
	
	def calc_w_0(self):
		w_0 = 0
		for j in range(0,len(self.alphas)):
                        inner_sum = 0
                        for i in range(0,len(self.alphas)):
                                inner_sum += self.alphas[i] * self.Y_train[i] * (np.dot(self.X_train[j].T, self.X_train[i]))
                        w_0 += (self.Y_train[j]-inner_sum)
                        #w_0 = self.Y_train[i] - val
                w_0 = w_0/len(self.alphas)
		self.w_0 = w_0
		return True
	
	def clean_alphas(self,alphas):
		#a_s = [el for el in alphas if el > .0000005]
		#return a_s
		return alphas
	
	def test(self, x_test, alphas, w_0, Y_train, X_train):
		#if self.kernel = self.rbf_kernel:
		#else:
		val = 0
		for i in range(0,len(alphas)):
			#print i	
			temp_1 = alphas[i] * Y_train[i]
			#print np.matrix(x_test).T.shape
			#print np.matrix(X_train[i]).shape
			
			temp_2 = (self.kernel(np.matrix(x_test), np.matrix(X_train[i]).T))
			#print temp_1.shape
			#print temp_2.shape
			val += temp_1 * temp_2
			#val += alphas[i] * Y_train[i] * (self.kernel(x_test.T, self.X_train[i]))
		val += w_0
		return val
		#print val
		'''decision function'''
		#if val > 0:
		#	return 1
		#elif val< 0:
		#	return -1
		#else:
		#	return 0

	def error(self):
		return True	
