import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sb
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x, w, b):
  return 1 / (1 + np.exp(-x*w-b))


def computeLoss(w, b, x, y):
	loss = 0
	
	# Loss = -(y*h(x) + (1-y)*(1-h(x)))
	for i in range(len(x)):
		loss += -(y[i]*np.log(sigmoid(x[i], w, b)+ 1e-20) + (1-y[i])*np.log(1 - sigmoid(x[i], w, b) + 1e-20))

	return loss


def computeLossL2(w, b, x, y):
	loss = np.zeros((len(w), len(w[0])))
	
	# Loss = -(y*h(x) + (1-y)*(1-h(x)))
	for i in range(len(x)):
		for j in range(len(w)):
			for k in range(len(w[0])):
				loss[j,k] += (y[i] - sigmoid(x[i], w[j,k], b[j,k]))**2
				# loss[j,k] += -(y[i]*np.log(sigmoid(x[i], w[j,k], b[j,k])) + (1-y[i])*np.log(1 - sigmoid(x[i], w[j,k], b[j,k])))

	return loss


def getLossSurface(x, y):
	# loss = []
	# for i in np.arange(-10, 10,0.2):
	# 	for j in np.arange(-10, 10, 0.2):
	# 		currentLoss = computeLossL2(i, j, x, y)
	# 		loss.append([i, j, currentLoss])

	# loss = np.array(loss)
	# return loss
	w = np.arange(-20, 20,0.1)
	b = np.arange(-20, 20,0.1)
	w_, b_ = np.meshgrid(w, b)
	loss = computeLossL2(w_, b_, x, y)

	return w_, b_, loss


def updateWeights(w, b, x, y, learningRate):
	dw, db = 0, 0
	N = len(x)

	# Compute gradients
	for i in range(len(x)):
		dw += (y[i] - sigmoid(x[i], w, b))*(x[i])
		db += (y[i] - sigmoid(x[i], w, b))

	w += (dw)*learningRate
	b += (db)*learningRate

	currentLoss = computeLoss(w, b, x, y)
	print("Weight: {}, bias: {}, loss: {}".format(w, b, currentLoss))

	return w, b, currentLoss


x = np.arange(0, 10,0.1)

x_data = [1, 2, 1.5, 2.2, 7, 6.7, 8.3, 7.5]
y_data = [0, 0, 0, 0, 1, 1, 1, 1]

fig = plt.figure()
plt.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')

# Compute loss at different values of parameters
w_, b_, lossValues = getLossSurface(x_data, y_data)

# Perform backprop and update weights
numIterations = 50
learningRate = 1e0
w, b = 15, 10
for i in range(numIterations):
	# Compute gradients and update weights
	w, b, loss = updateWeights(w, b, x_data, y_data, learningRate)
	y = sigmoid(x, w, b)
	plt.plot(x, y, c=np.random.rand(3,))
	# plt.plot(w, b, c='r')
	# ax2.scatter(w, b, loss, c='r')
	plt.pause(0.05)


# fig = plt.figure()
# cp = plt.plot(w_, b_, lossValues)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(w_, b_, lossValues, cmap='terrain')
ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.set_zlabel('loss')


plt.show()