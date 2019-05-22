import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sb
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def convergence(x1, x2, y, w):
	misClassifications = 0
	for i in range(len(x1)):
		# print("x1: {}, x2: {}".format(x1[i], x2[i]))
		output =  w[0] + x1[i]*w[1] + x2[i]*w[2]
		if y[i] == 1 and output < 0:			
			misClassifications += 1
		elif y[i] == 0 and output >= 0:
			misClassifications += 1		

	print("Number of misclassifications: {}".format(misClassifications))
	if misClassifications == 0:
		return True

	return False

# Data
x1 = [1.2, 2, 0.5, 0.8, 4.4, 5.6, 7, 4.3]
x2 = [2.3, 1.9, 2, 0.9, 5.6, 9, 8.8, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Weight parameters
w = np.array([1e-4, 1e-4, 1e-4])

# print(convergence(x1, x2, y, w))

noIter = 0
while not convergence(x1, x2, y, w):
	i = np.random.randint(low=1, high=len(x1))
	output =  w[0] + x1[i]*w[1] + x2[i]*w[2]
	print("Iter: {}, Weights: {}".format(noIter, w))
	if y[i] == 1:		
		if output < 0:
			w[0] += 1
			w[1] += x1[i]
			w[2] += x2[i]
			

	else:
		if output >= 0:
			w[0] -= 1
			w[1] -= x1[i]
			w[2] -= x2[i]
			
	noIter += 1	
	


x1p = np.linspace(0,10,100).reshape(-1,1)
x2p = - w[1]/w[2]*x1p - w[0]/w[2]

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(x1[:4], x2[:4], label='C1')
ax2.scatter(x1[4:], x2[4:], label='C0')
plt.plot(x1p, x2p, c = 'k', linewidth = 2, label = 'perceptron')
plt.xlim([0, 10])
plt.ylim([0, 12])
plt.legend(loc = 1, fontsize = 10)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

plt.show()