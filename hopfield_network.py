import numpy as np


class HopfieldNetwork:
    def __init__(self):
        self.W = None

    def train(self, X: np.ndarray):
        M, N = len(X), len(X[0])
        self.W = (1 / N) * X.dot(X.T)
        # np.fill_diagonal(self.W, 0)

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
            indices = np.random.permutation(indices)
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


x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]

X = np.array([x1, x2, x3]).T
model = HopfieldNetwork()
model.train(X)

# all possible pattern 8-neuron
N = 8
k = 2 ** N
print('2^N=', k)
rp = np.ones([k, N])

for i in range(k):
    l = len(bin(i)) - 2
    for j in range(l):
        if bin(i)[j + 2] == '1':
            rp[i, N - l + j] = -1

n = 0
for i in range(k):
    A = rp[i]
    B = model.update(A)
    if (A == B).all():
        n += 1
        print('attractor', n, ':', A)
