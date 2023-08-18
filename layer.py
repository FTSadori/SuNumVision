import numpy as np


class Layer:
    rand_range = 5

    def __init__(self, prev_size, this_size):
        self.W = np.random.uniform(low=-self.rand_range,
                                   high=self.rand_range,
                                   size=(this_size, prev_size))
        self.bias = np.random.uniform(low=-self.rand_range,
                                      high=self.rand_range,
                                      size=this_size)
        self.a = []

    def calculate(self, a_prev, func):
        self.a = func(np.dot(self.W, a_prev) + self.bias)
        return self.a
