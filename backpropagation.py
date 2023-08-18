import numpy as np


class GLayer:
	def __init__(self, prev_size, layer_size):
		self.W = np.zeros(shape=(layer_size, prev_size))
		self.bias = np.zeros(shape=(layer_size, 1))


class Gradient:
	def __init__(self, sizes):
		self.layers = [GLayer(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))]
