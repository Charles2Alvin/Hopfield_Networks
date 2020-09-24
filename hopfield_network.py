import numpy as np
from matplotlib import pyplot as plt


class HopfieldNetwork:
    def __init__(self):
        self.W = None

    def train(self, X: np.ndarray):
        M, N = len(X), len(X[0])
        self.W = (1 / N) * X.dot(X.T)
        np.fill_diagonal(self.W, 0)

    def print_W(self):
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                print("%.1f  \t" % (self.W[i][j]),  end='')
            print()

    def update(self, p_start):
        """
        Apply the update rule on the input pattern V to get the restored pattern
        which is an attractor in the network's storage
        :param p_start: the start pattern
        :return: the restored pattern
        """
        V = np.copy(p_start)
        m = len(V)
        indices = [i for i in range(m)]
        iter = 1
        while True:
            cnt = 0
            for i in indices:
                value_old = V[i]
                value_new = np.sign(self.W[:, i].dot(V))
                if value_new != value_old:
                    V[i] = value_new
                    cnt += 1
                # print(i, value_old, value_new, self.W[:, i].dot(V))
            if cnt == 0:
                # print("Converged in %s iterations" % iter)
                break
            iter += 1

        return V


f = open("pict.dat")
raw_data = f.read().split(',')
length = len(raw_data)
N = int(length / 1024)
data = np.zeros((1024, N))
for i in range(length):
    raw_data[i] = float(raw_data[i])
data = np.array(raw_data).reshape((N, 1024))
p1, p2, p3 = data[0], data[1], data[2]

X = np.array([p1, p2, p3]).T
model = HopfieldNetwork()
model.train(X)
rp1 = model.update(p1)
print((rp1 == p1).all())