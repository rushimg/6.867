from scipy import signal
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize

def l2_dist(x1,x2):
	#dist = sum((x1-x2)**2)
	dist = 0
	m = np.shape(x1)[0]
	for i in range(0,m):
		dist += (x1[i] - x2[i])*(x1[i] - x2[i])
	return math.sqrt(dist)

def loss(a,b):
	return a-b

def gradie2ntDescent(x, y, theta, alpha, numIterations):
	#print x.shape
	#print theta.shape
	#m,n = x.shape
	xTrans = x.transpose()
	for i in range(0, numIterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		# avg cost per example (the 2 in 2*m doesn't really matter here.
		# But to be consistent with the gradient, I include it)
		cost = np.sum(loss ** 2) / (2 * m)
		print("Iteration %d | Cost: %f" % (i, cost))
		# avg gradient per example
		gradient = np.dot(xTrans, loss) / m
		# update
		theta = theta - alpha * gradient
	return theta

def gradientDescent(x ,y , initial_guess, objective_func, gradient_func, step_size, convergence):
        count_iter = 0 # keep track of number of iterations
	#x = np.array(initial_guess)
        print x.shape
	xTrans = x.transpose()
	m,n = x.shape
	theta = np.ones(n)
	#print theta.shape
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
		print theta
        return theta,count_iter

def genGaussian(numPoints, mu, sigma):
	x = np.zeros(shape=(numPoints, 1))	
	y = np.zeros(shape=numPoints)
	for i in range(0,numPoints ):
			
		x[i][0] = i
      
      		y[i] = -np.exp(-np.power(x[i][0] - mu, 2.) / 2 * np.power(sigma, 2.))
	return x, y

def gradient_analytical(xTrans,h0,y):
	return np.dot(xTrans, h0-y)

def gaussian_1d(x):
        mu = 5
	sigma = 1
	return -np.exp(-np.power(x - mu, 2.) / 2 * np.power(sigma, 2.))

def genRosenbrock(numPoints):
	print numPoints
	x = np.zeros(shape=(numPoints, 2))
	y = np.zeros(shape=numPoints)
        for i in range(0,numPoints):
		x[i][0] = i
		x[i][1] = 2
                y[i] = .5*(1 - x[i][0])**2 + (2 - x[i][0]**2)**2
	return x, y
def rosenbrock(x):
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

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

'''
x_list = []
y_list = []
for x in range(-100,100):
	x_list.append(x)
	y_list.append(gaussian_1d(x))
#plt.plot(x_list,y_list)
#plt.show()
x = np.array([el for el in x_list])
y = np.array([el for el in y_list])
'''
#x, y = genData(100, 25, 10)
#x, y = genGaussian(100, 5, 1)
#m, n = np.shape(x)
#initial_guess = np.ones(n)
#objective_func = gaussian_1d
#convergence = .000001
#x, y = genRosenbrock(100)
#m, n = np.shape(x)
#plt.plot(x, y)
#plt.show()

alpha = 0.01
objective_func = rosenbrock
x, y = genData(100, 25, 10)
m, n = np.shape(x)
theta = np.ones(n)
initial_guess = [1,1]
convergence = .0001
#gradientDescent(x, y, theta, alpha, 1000000)

theta_min, iters = gradientDescent(x, y, initial_guess, loss, gradient_analytical, alpha, convergence)
print "RMG implementation"
print "Minima : " + str(theta_min) + " convereged at " + str(iters) + " iterations " 

print "scipy.optimize BFGS"
print optimize.fmin_bfgs(objective_func, initial_guess)


