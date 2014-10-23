from scipy import signal
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize

def l2_dist(x1,x2):
	dist = 0
	m = np.shape(x1)[0]
	for i in range(0,m):
		dist += (x1[i] - x2[i])*(x1[i] - x2[i])
	return math.sqrt(dist)

def loss(a,b):
	return a-b


def gradientDes2cent(x, y, theta, alpha, numIterations):
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

def gradientDescent(x, y, objective_func, gradient_func, theta, step_size, convergence):
	#print x.shape
	#print theta.shape
	m, n = np.shape(x)
	xTrans = x.transpose()
	count_iter = 0 # keep track of number of iterations
	# Initialiize theta_new to some large value
	#dims = np.shape(theta)[0]
	#theta_new=np.empty(dims)
	#theta_new.fill(1000000)
	#theta= np.ones(m)
	error = 1
	while abs(error) > convergence:
		print error
		#print theta
		count_iter+=1

		h0 = np.dot(x, theta) # h0
		#loss = objective_func(hypothesis,y)
		loss = h0 - y
		#print loss
		#cost = np.sum(loss ** 2) / (2 * m)	
		#print cost
		#print("Iteration %d | Cost: %f" % (count_iter, cost))
		#print("Iteration %d" % (count_iter))
		# avg gradient per example
		#gradient = np.dot(xTrans, loss)/m
		# update
		#theta_new = theta - step_size *gradient
		theta_new = theta - step_size * gradient_func(xTrans,h0,y)/m
		error = l2_dist(theta_new,theta)
		theta = theta_new
	return theta,count_iter

def gradient_analytical(xTrans,h0,y):
	return np.dot(xTrans, h0-y)

def normal_2d():
	R = np.random.multivariate_normal([4, 2], [[1, 1.5], [1.5, 3]], 1000)	
	return R

def gaussian_1d(mu, sigma, num_points):
	x = np.linspace(mu-5,mu+5,num_points)
	y = -np.exp(-np.power(x - mu, 2.) / 2 * np.power(sigma, 2.))
	#one = np.ones(y.shape)
	#y += one
	#y += one
	#y += one
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
def gauss(x):
	mu = 5
	sigma = 1
	y = -np.exp(-np.power(x - mu, 2.) / 2 * np.power(sigma, 2.))
	return y
def genRosenbrock(numPoints):
        print numPoints
        x = np.zeros(shape=(numPoints, 2))
        y = np.zeros(shape=numPoints)
        for i in range(0,numPoints):
                x[i][0] = i
                x[i][1] = 2
                y[i] = .5*(1 - x[i])**2 + (2 - x[i]**2)**2
        return x, y

def rosenbrock(x):   # The rosenbrock function
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def linear(x):
	y = x + 30
	return y

num_points = 100
x, y = gaussian_1d(5,1,num_points)
#print y
plt.plot(x,y)
plt.show()
theta = np.ones(1)
#theta = np.array([2.82552875])


# original line
x = []
#x[0] = np.linspace(0,100,100)
#x[1] = np.linspace(0,100,100)
#x, y = genRosenbrock(100)
x, y = genData(100, 25, 10)
#plt.plot(x,y)
#plt.show()
#print y.shape
#theta = np.ones(2)

m, n = np.shape(x)
#theta = np.on
#print gaussian_1d(1)
initial_guess = [0]
theta = np.ones(n)
convergence = .00001
alpha = 0.00005
objective_func = linear
#theta_min, iters = gradientDescent(rosenbrock, [2,2], loss, gradient_analytical, alpha, convergence)
theta_min, iters = gradientDescent(x, y, loss, gradient_analytical, theta, alpha, convergence)
print "Minima : " + str(theta_min) + " convereged at " + str(iters) + " iterations " 
print "scipy.optimize BFGS"
print optimize.fmin_bfgs(objective_func, initial_guess)

