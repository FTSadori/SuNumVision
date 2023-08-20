import numpy as np
import backpropagation as bp
import json as js


class Layer:
    random_min = -1
    random_max = 1

    def __init__(self, prev_layer_size, this_layer_size):
        self.W = np.random.uniform(-1,
                                   1,
                                   size=(this_layer_size, prev_layer_size))
        self.b = np.random.uniform(-1,
                                   1,
                                   size=(this_layer_size, 1))
        self.z = np.zeros(shape=(this_layer_size, 1))

    def calculate(self, a0, fnc):
        self.z = np.dot(self.W, a0) + self.b
        return fnc(self.z)


class NeuralNetwork:
    def __init__(self, l_sizes):
        self.sizes = l_sizes
        self.layers = [Layer(l_sizes[i - 1], l_sizes[i]) for i in range(1, len(l_sizes))]
        self.G = bp.Gradient(l_sizes)
        self.g_count = 0
        self.a = [np.zeros(shape=(size, 1)) for size in self.sizes]

    def calculate(self, a0, fnc, last_layer=None):
        if len(self.a[0]) != len(a0):
            raise Exception("a0 must be " + str(len(self.a[0])) + " elements long")
        self.a[0] = a0
        for i in range(len(self.layers)):
            self.a[i + 1] = self.layers[i].calculate(self.a[i], fnc)
        return self.a[-1]

    def main_propagation_cycle(self, y, der_fnc, der_cost):
        dal = der_cost(self.a[-1], y)

        for L in reversed(range(len(self.layers))):
            o = der_fnc(self.layers[L].z) * dal

            self.G.layers[L].bias += o

            w = np.zeros(shape=(len(self.a[L]), len(self.a[L + 1])))
            for k in range(len(self.a[L])):
                w[k] = (np.transpose(self.a[L][k] * o)[0])
            self.G.layers[L].W += np.transpose(w)

            if L != 0:
                n_dal = []
                for k in range(len(self.a[L])):
                    n_dal.append(sum(np.transpose(self.layers[L].W)[k] * np.transpose(o)[0]))
                dal = np.transpose([n_dal])

    def to_file(self, file_path):
        f = open(file_path, 'w')
        f.write(js.dumps(self.to_dict()))
        f.close()

    def from_file(self, file_path):
        f = open(file_path, 'r')
        dct = js.loads(f.readline())
        for num in dct:
            self.layers[int(num)].W = np.array(dct[num]['W'])
            self.layers[int(num)].b = np.array(dct[num]['b'])

    def back_propagation(self, max_g_count, y, der_fnc, der_cost):
        self.main_propagation_cycle(y, der_fnc, der_cost)

        self.g_count += 1
        if self.g_count == max_g_count:
            self.g_count = 0
            for LL in range(len(self.layers)):
                self.layers[LL].W -= self.G.layers[LL].W / max_g_count
                self.layers[LL].b -= self.G.layers[LL].bias / max_g_count

            self.G = bp.Gradient(self.sizes)

    def instant_back_propagation(self, rate, y, der_fnc, der_cost):
        self.main_propagation_cycle(y, der_fnc, der_cost)

        for LL in range(len(self.layers)):
            self.layers[LL].W -= self.G.layers[LL].W * rate
            self.layers[LL].b -= self.G.layers[LL].bias * rate

        self.G = bp.Gradient(self.sizes)

    def to_dict(self):
        dc = {}
        for i in range(len(self.layers)):
            dc[i] = {'W': self.layers[i].W.tolist(), 'b': self.layers[i].b.tolist()}
        return dc
