import numpy as np
from matplotlib import pyplot as plt


class HopfieldNetwork:
    def __init__(self):
        self.W = None

    def train(self, X: np.ndarray, p_avg=0):
        M, N = len(X), len(X[0])
        avg = np.zeros((M, N))
        avg.fill(p_avg)
        self.W = (1 / N) * (X - avg).dot((X - avg).T)
        np.fill_diagonal(self.W, 0)

    def print_W(self):
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                print("%.1f  \t" % (self.W[i][j]),  end='')
            print()

    def update_async(self, p_start, mode: str = 'sequential', log: list = None):
        """
        Apply the update rule on the input pattern V to get the restored pattern
        which is an attractor in the network's storage
        :param mode:
        :param p_start: the start pattern
        :param log: energy log array to be filled
        :return: the restored pattern
        """
        V = np.copy(p_start)
        m = len(V)
        indices = [i for i in range(m)]
        iter = 1
        while True:
            if log is not None:
                e = self.energyOf(V)
                log.append(e)
            cnt = 0
            if mode == 'random':
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
            if iter > 1e3:
                print("Reached maximum iteration", iter)
                break
            iter += 1

        return V

    def update_sync(self, p_start, log: list = None):
        """
        Apply the update rule on the input pattern V to get the restored pattern
        which is an attractor in the network's storage
        :param p_start: the start pattern
        :param log: energy log array to be filled
        :return: the restored pattern
        """
        V = np.copy(p_start)
        iter = 1
        while True:
            if log is not None:
                e = self.energyOf(V)
                log.append(e)
            V_new = np.sign(self.W.dot(V))
            V = V_new
            diff = np.sum(np.abs(V - V_new))
            if diff == 0:
                break
            iter += 1

        return V

    def update_slightly(self, p_start, bias, log: list = None):
        V = np.copy(p_start)
        N = len(V)
        theta = np.zeros(N)
        theta.fill(bias)
        base = np.zeros(N)
        base.fill(0.5)
        iter = 1
        while True:
            if log is not None:
                e = self.energyOf(V)
                log.append(e)
            V_new = base + 0.5 * sign(self.W.dot(V) - theta)
            V = V_new
            diff = np.sum(np.abs(V - V_new))
            if diff == 0:
                break
            iter += 1

        return V

    def energyOf(self, x):
        """
        Compute the energy of the given state
        :param x: a n-by-1 array pattern which represents a state
        :return: a float value
        """
        # take it as a quadratic form
        return - x.dot(self.W).dot(x)

    @staticmethod
    def plot_energy(energy_log):
        for i in range(len(energy_log)):
            plt.scatter(i, energy_log[i])
        plt.plot(energy_log)
        plt.title("Energy changes")
        plt.show()


def add_noise(arr: np.ndarray, ratio: float):
    """
    Randomly flip a selected number of units
    :param arr: the array to add noise
    :param ratio: the ratio of noise
    :return: the array with noise
    """
    p = np.copy(arr)
    n = len(p)
    permutation = [i for i in range(n)]
    m = int(n * ratio)
    np.random.seed(2)
    indices = np.random.choice(permutation, m)
    for index in indices:
        p[index] = - p[index]

    return p


def read_data(file_name):
    f = open(file_name)
    raw_data = f.read().split(',')
    length = len(raw_data)
    N = int(length / 1024)
    for i in range(length):
        raw_data[i] = float(raw_data[i])
    data = np.array(raw_data).reshape((N, 1024))

    return data


def repair_degree(p1, p2):
    return 1 - 0.5 * np.sum(np.abs(p1 - p2)) / len(p1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sign(nums: np.ndarray):
    for i in range(nums.shape[0]):
        nums[i] = 1 if nums[i] >= 0 else -1

    return nums


def sparse_pattern(N, P, p):
    """
    Generate sparse patterns
    :param N: the number of units
    :param P: the number of patterns
    :param p: the activity
    :return: sparse pattern matrix
    """
    ones = np.ones(int(p * N))
    zeros = np.zeros(N - int(p * N))
    base = np.concatenate((ones, zeros))
    X = []
    for i in range(P):
        np.random.seed(P + i)
        X.append(np.random.permutation(base))
    X = np.array(X).T

    return X


N, P = 100, 20
model = HopfieldNetwork()
X = sparse_pattern(N, P, 0.1)
model.train(X)

cnt_list = []
theta_list = np.linspace(0, 1, 11)
for theta in theta_list:
    cnt = 0
    accuracy = []
    for i in range(P):
        x = X[:, i]
        rp = model.update_slightly(x,  bias=theta)
        accuracy.append(repair_degree(x, rp))
        cnt += 1 if (x == rp).all() else 0
    cnt_list.append(cnt)
    print("bias term theta = %.2f, stored patterns = %s" % (theta, cnt))


