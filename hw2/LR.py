import numpy as np
import scipy.optimize
import random
import math
class LR:
	def __init__(self, X, Y, L):
		self.X_train = X
		self.Y_train = Y 
		# L is lambda
		self.l = L

	#linear kernel
	def linear_kernel(self,X,X_prime):
		return np.dot(X,X_prime)

	# rbf kernel 	
	def rbf_kernel(x, y, sigma=5.0):
	    	return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

	def train_gold(self):
		''' sklearn imlplementation to check results against '''
		from sklearn import svm
		y = self.Y_train
                X = self.X_train
		model = svm.LinearSVC(C=self.C)
		return model.fit(X,y)
	
	def test_gold(self,x_test,model):
		return model.predict(x_test)

	def NLL(self,args):
		#alpha = np.matrix(args[0]).T 
		alpha = np.matrix(args[0:-1]).T
		w_0 = args[-1]
		#w_0 = args[1]
		i = 0
		summation = 0
		Y = self.Y_train
		X = self.X_train
		for el in Y:
			#print np.matrix(alpha).shape
			#print np.matrix(X[i]).shape
			#print X.T.shape
			#print np.matrix(X[i])*X.T*np.matrix(alpha).T
			summation += math.log(1+math.exp(-Y[i]*(np.matrix(X[i])*X.T*alpha+w_0)))
			i += 1
		# add in regularization term
		print np.sum(alpha)
		regularization = self.l * np.sum(alpha)
		summation += regularization
		return summation

	def train(self):
		Y = self.Y_train 
		X = self.X_train
		
		#args = list()
		#args.append(np.zeros(len(self.Y_train)))
		#args.append(0)
		args = np.zeros(len(self.Y_train)+1)
		optimal_values = scipy.optimize.fmin_bfgs(self.NLL, args)
		print optimal_values
		'''
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
    			for j in range(n_samples):
	        		K[i,j] = self.linear_kernel(X[i], X[j])
		'''
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
	
	def test(self, x_test):
		val = 0
		for i in range(0,len(self.alphas)):
			#print i	
			val += self.alphas[i] * self.Y_train[i] * (np.dot(x_test.T, self.X_train[i]))
		val += self.w_0
		'''decision function'''
		if val > 0:
			return 1
		elif val< 0:
			return -1
		else:
			return 0

	def error(self):
		return True	
