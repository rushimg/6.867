import numpy as np
import random
import matplotlib.pyplot as plt
import math
def l2_dist(x1,x2):
        dist = 0
        m = np.shape(x1)[0]
        for i in range(0,m):
                dist += (x1[i] - x2[i])*(x1[i] - x2[i])
        return math.sqrt(dist)

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, convergence):
	xTrans = x.transpose()
	error = 1
	i = 0
	while abs(error) > convergence:
		
    #for i in range(0, numIterations):
		i += 1 
		hypothesis = np.dot(x, theta)
        	loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        	cost = np.sum(loss ** 2) / (2 * m)
        	print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        	gradient = np.dot(xTrans, loss) / m
        # update
	
        	theta_new = theta - alpha * gradient
    		error = l2_dist(theta_new,theta)
		theta_new = theta
		print error
	return theta
'''
def gradientDescent(x ,y , initial_guess, objective_func, gradient_func, step_size, convergence):
        count_iter = 0 # keep track of number of iterations
        #x = np.array(initial_guess)
        #print x.shape
        #print 
        xTrans = x.transpose()
        m, n = x.shape
        theta = np.ones(m)
        error = 1
        while abs(error) > convergence:
                print theta
                count_iter+=1
                h0 = np.dot(x, theta) # h0
                #y = func(x)
                #loss = h0 - y
                theta_new = theta - step_size * gradient_func(xTrans,h0,y)/m
                error = l2_dist(theta_new,theta)
                theta = theta_new
        return theta,count_iter
'''
def gradient_analytical(xTrans,h0,y):
        return np.dot(xTrans, h0-y)

def genGaussian(numPoints, mu, sigma):
        offset = 1
	x = np.zeros(shape=(numPoints, 1))
        y = np.zeros(shape=numPoints)
        for i in range(0,numPoints ):
                x[i][0] = i
	
                y[i] = -np.exp(-np.power(x[i] - mu, 2.) / 2 * np.power(sigma, 2.))
        return x, y

def gaussian_1d(mu, sigma, num_points):
        x = np.linspace(mu-5,mu+5,num_points)
        y = -np.exp(-np.power(x - mu, 2.) / 2 * np.power(sigma, 2.))
        one = np.ones(y.shape)
        y += one
        y += one
        y += one
        #y = y.reshape((num_points,1))
        x = x.reshape((num_points,1))
        return x,y

def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
#x, y = genData(100, 25, 10)
#x, y = genGaussian(100, 20, 20)
x, y = gaussian_1d(5,1,100)
m, n = np.shape(x)
plt.plot(x, y)
plt.show()
convergence = .0001
alpha = 0.0001
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, convergence)
#theta = gradientDescent(x ,y , theta, None, gradient_analytical, alpha, convergence)
print(theta)
