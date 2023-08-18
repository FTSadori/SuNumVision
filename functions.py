import numpy as np
from PIL import Image
import math as mt


def sig(x):
	return 1 / (1 + np.exp(-x))

def dersig(x):
	return sig(x) * (1 - sig(x))

def ReLU(x):
	ans = np.zeros(shape=(len(x), 1))
	for j in range(len(x)):
		ans[j][0] = max(0, x[j])
	return ans

def derReLU(x):
	ans = np.zeros(shape=(len(x), 1))
	for j in range(len(x)):
		ans[j][0] = 1 if x[j] > 0 else 0
	return ans

def tanh(x):
	ans = np.zeros(shape=(len(x), 1))
	for j in range(len(x)):
		ans[j][0] = mt.tanh(x[j][0])
	return ans

def dertanh(x):
	ans = np.zeros(shape=(len(x), 1))
	for j in range(len(x)):
		ans[j][0] = 1 - mt.tanh(x[j]) ** 2
	return ans

def softmax(x):
	x = x.flatten()
	s = sum(mt.exp(n) for n in x)
	ans = np.zeros(shape=(len(x), 1))
	for j in range(len(x)):
		ans[j][0] = mt.exp(x[j]) / s
	return ans
