import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sb
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def computeLoss(beta_0, beta_1, x, y):
	loss = 0
	
	# Loss = (y - beta_0 * x - beta_1)**2
	for i in range(len(x)):
		loss += (y[i] - beta_0*x[i] - beta_1)**2

	return loss/len(x)


def getLossSurface(x, y):
	loss = []
	for i in np.arange(-10, 10,0.5):
		for j in np.arange(-10, 10, 0.5):
			currentLoss = computeLoss(i, j, x, y)
			loss.append([i, j, currentLoss])

	loss = np.array(loss)
	return loss


def updateWeights(beta_0, beta_1, x, y, learningRate):
	db0, db1 = 0, 0
	N = len(x)

	# Compute gradients
	for i in range(len(x)):
		db0 += -2*(y[i] - beta_0*x[i] - beta_1)*(x[i])
		db1 += -2*(y[i] - beta_0*x[i] - beta_1)

	beta_0 -= (db0/N)*learningRate
	beta_1 -= (db1/N)*learningRate

	currentLoss = computeLoss(beta_0, beta_1, x, y)
	print("Weight: {}, bias: {}, loss: {}".format(beta_0, beta_1, currentLoss))

	return beta_0, beta_1, currentLoss

# Link: https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
def drawLine(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')


# Load the tips dataset
df = sb.load_dataset('tips')

# Convert to numpy
dataset = df.to_numpy()

amount = dataset[:,0]
tips = dataset[:,1]

# Compute loss at different values of parameters
lossValues = getLossSurface(amount, tips)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(lossValues[:,0], lossValues[:,1], lossValues[:,2])
ax2.set_xlabel('beta_0')
ax2.set_ylabel('beta_1')
ax2.set_zlabel('loss')

# Perform backprop and update weights
numIterations = 100
learningRate = 1e-4
beta_0, beta_1 = 10, 0
for i in range(numIterations):
	# Compute gradients and update weights
	beta_0, beta_1, loss = updateWeights(beta_0, beta_1, amount, tips, learningRate)
	ax2.scatter(beta_0, beta_1, loss, c='r')
	plt.pause(0.05)

fig = plt.figure()
plt.scatter(amount, tips, s=2)
plt.xlabel('Amount')
plt.ylabel('Tips')
drawLine(beta_0, beta_1)


plt.show()