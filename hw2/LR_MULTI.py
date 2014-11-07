import numpy as np

import math
class LR:
	def __init__(self, X, Y, L):
		self.X_train = X
		self.Y_train = Y 
		self.l = L
	
	#linear kernel
	def linear_kernel(self,X,X_prime):
		return np.dot(X,X_prime)
	
	def rbf_kernel(x, y, sigma=5.0):
	    	return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

	def train_gold(self):
                ''' sklearn imlplementation to check results against '''
                import sklearn
		from sklearn import linear_model

                y = self.Y_train
                X = self.X_train
                model = sklearn.linear_model.LogisticRegression()
                return model.fit(X,y)

        def test_gold(self,x_test,model):
                return model.predict(x_test)
	
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

	def NLL(self,args):
		args = args.reshape(self.Y_train.shape[0], 4)	
		#print np.matrix(args).shape
		X = self.X_train
		Y = self.Y_train
		alpha = args[:,:-1]
		#w_0 = 0
		w_0 = args[:,-1]
		summ = 0 
		i = 0
	
		for el in (self.Y_train):
			#temp_1 = np.dot(self.K,alpha)
			#print 'temp1' , temp_1.shape
			#print Y[i].shape	
			#temp_2 = Y[i] * np.dot(s
			exponent = np.sum(-1*Y[i]*np.dot(self.K,alpha).T)
			summ += math.log(1+math.exp(exponent))
			#summ += math.log(1+math.exp(-1*Y[i]*(np.dot(np.dot(X[i],X.T),alpha)+w_0)))
			#kernel = np.dot(X[i],X.T)
                        #inside = np.dot(kernel, alpha) + w_0
                        ##print inside
			#summ += math.log(1+math.exp(np.dot(-1*Y[i],(inside))))
                        #i += 1
		#summ += self.l*np.sum(alpha)
			#summ += math.log(1+math.exp(-1*self.Y_train[i]))
	
		return summ
	
	def train(self):
		from scipy.optimize import fmin_bfgs
		Y = self.Y_train 
		X = self.X_train
		n_samples = self.Y_train.shape[0]
		
		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
    			for j in range(n_samples):
	        		K[i,j] = self.linear_kernel(X[i], X[j])
		self.K = K
		args = np.zeros((n_samples, 4))
		#[0.1]*(n_samples+1)
		#args = np.ones((n_samples+1))*.001
		solution = fmin_bfgs(self.NLL, args)

		self.alphas = solution[:,0:-1]
		print self.alphas
		self.w_0 = solution[:,-1]
		self.calc_w_0()
		#print self.w_0
		num_alphas = 0
		for a in self.alphas:
			if a > 1e-5:
				num_alphas += 1
		print 'num_alphas' ,num_alphas

			
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

