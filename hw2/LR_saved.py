import numpy as np
import scipy.optimize
import random
import math
import sklearn 

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
		from sklearn import linear_model
		
		y = self.Y_train
                X = self.X_train
		model = sklearn.linear_model.LogisticRegression()
		return model.fit(X,y)
	
	def test_gold(self,x_test,model):
		return model.predict(x_test)

	def sig(self, t):
		return 1/(1+math.exp(t))

	def NLL(self,args):
		alpha = np.matrix(args[0:-1]).T
		#W = np.matrix(args).T
		#W = np.matrix(args[0:-1])
		#print X[0].shape
		w_0 = np.matrix(args[-1])
		i = 0
		summation = 0
		Y = self.Y_train
		X = self.X_train
		'''
                K = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                        for j in range(n_samples):
                                K[i,j] = self.linear_kernel(X[i], X[j])
		'''
		i = 0 
		summation = 0
		for el in Y:	
			#print np.matrix(alpha).shape
			#print np.matrix(X[i]).shape
			#print X.T.shape
			#print np.matrix(X[i])*X.T*np.matrix(alpha).T
			#summation += math.log(1+math.exp(-Y[i]*(X[i]*W+w_0)))
			#print K.shape
			#print (X[i]*X.T).shape
			#inside = K*alpha+w_0
			W = X.T * alpha
			inside = np.dot(X[i], W) + w_0
			summation += math.log(1+math.exp(-Y[i]*(inside)))
			i += 1
		# add in regularization term
		#regularization = self.l * np.sum(abs(alpha))
		#summation += regularization
		return summation

	def train(self):
		Y = self.Y_train 
		X = self.X_train
		#args = np.zeros(3) 
		#print w_h_i
		#w_h = w_h_i[-1]

		args = np.zeros(len(self.Y_train)+1) 
		optimal_values = scipy.optimize.fmin_bfgs(self.NLL, args)
		self.alphas = optimal_values[0:-1]
		#self.alphas = np.array([.531, 0, .367, .163])
		#print 'alphas', self.alphas
		w = X.T*np.matrix(self.alphas).T
		print 'w', w
		print 'alphas', self.alphas
		self.w_0 = optimal_values[-1]
		#self.w_0 = -1.21
		'''
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
    			for j in range(n_samples):
	        		K[i,j] = self.linear_kernel(X[i], X[j])
		'''
		# clean alphas
		#temp = list()	
		#for a in self.alphas:
			#if abs(a) > 1e-5:
			#	temp.append(a)
		#self.alphas = np.asarray(temp)
		print 'num_alphas' , len(self.alphas)

		print 'w_0' , self.w_0
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
		#print val
		'''decision function'''
		if val > 0:
			return 1
		elif val< 0:
			return -1
		else:
			return 0

	def error(self):
		return True	
