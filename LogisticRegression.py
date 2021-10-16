import numpy as np
from metrics import *


class LogisticRegression(object):

    def __init__(self, add_bias=True,
                 learning_rate=1e-4,
                 epsilon=1e-4,
                 max_epoch=1000,
                 verbose=False,
                 batch_size=1,
                 momentum=0,
                 reset_each_time = True
                 ):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # to get the tolerance for the norm of gradients
        self.max_epoch = max_epoch  # maximum number of iteration of gradient descent
        self.verbose = verbose
        self.batch_size = batch_size
        self.momentum = momentum
        self.theta = None
        self.is_converged = False
        self.gradient_history = []
        self.accuracy_history = []
        self.reset_each_time = reset_each_time
    def reset(self):

        self.theta = None
        self.is_converged = False
        self.gradient_history = []
        self.accuracy_history = []




    def gradient(self, x, y):
        logistic = lambda z: 1. / (1 + np.exp(-z))  # logistic function
        N, D = x.shape
        yh = logistic(np.dot(x, self.theta))  # predictions  size N
        grad = np.dot(x.T, yh - y) / N  # divide by N because cost is mean over N points
        return grad

    def split_data(self, x, y):
        result = []
        data_size = self.batch_size
        if self.batch_size >= x.shape[0]:
            return [(x, y)]
        else:
            # starter = 0
            for i in range(0, x.shape[0], data_size):
                if i + data_size <= x.shape[0]:
                    cur_data = (x[i:i + data_size], y[i:i + data_size])
                    result.append(cur_data)
                else:
                    cur_data = (x[i:], y[i:])
                    result.append(cur_data)
        return result

    def shuffle(self, x, y):
        #         print("x shape {}" .format(x.shape))
        #         print("y shape {}".format(y.shape))
        #         data = np.stack((x, y), axis=1)
        #         # data.shape
        #         np.random.shuffle(data)
        #         new_x = data[:, :-1]
        #         new_y = data[:, -1]
        #         print("new_x shape {}".format(new_x.shape))
        #         print("new_y shape {}".format(new_y.shape))
        #         return new_x, new_y
        #         d = np.append(x, y if y.ndim > 1 else y[:, np.new_axis], axis = 1)
        #         np.random.shuffle(d)
        #         return d[:, :-1], d[:, -1]
        shuffle = np.random.permutation(x.shape[0])
        new_x = x[shuffle]
        new_y = y[shuffle]
        return new_x, new_y

    def fit(self, x, y):
        if self.reset_each_time:
            self.reset()
        # mini-batch ->
        logistic = lambda z: 1. / (1 + np.exp(-z))  # logistic function
        if x.ndim == 1:
            x = x[:, None]

        # print(f'GSLOLOKOKO: {x.shape}\n')

        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        #     add bias

        N, D = x.shape
        # print(f'GUTEN TAG: {(N, D)}\n')
        self.theta = np.zeros(D)
        cur_gradient = np.inf
        # inital_gradient
        num_of_iter = 0
        # number of iterations
        # the code snippet below is for gradient descent
        # get the batched data from split_data
        init_acc = self.accuracy(x, y)
        self.accuracy_history.append(init_acc)

        if self.verbose:
            print("current batch_size = {}".format(self.batch_size))
        epoch = 1
        while np.linalg.norm(cur_gradient) > self.epsilon and epoch < self.max_epoch:
            # stopped at loss < epsilon -> converged = True
            # if num_of_iter > self.max_iters -> converged = False
            #             new_x, new_y = x, y
            new_x, new_y = self.shuffle(x, y)
            # everytime go over the whole dataset, reshuffle the dataset once
            batched_data = self.split_data(new_x, new_y)
            # [(x, y),(x, y),(x, y),(x, y),(x, y)......(x, y)] according to batching
            batched_data_entries = len(batched_data)

            if self.verbose:
                print("start epoch {}".format(epoch))

            for i in range(batched_data_entries):
                # go over the whole dataset once according to the batch_size
                (batched_x, batched_y) = batched_data[i]

                cur_gradient = self.gradient(batched_x, batched_y)
                self.gradient_history.append(cur_gradient)

                if not self.momentum:
                    self.theta = self.theta - self.learning_rate * cur_gradient
                else:
                    b = self.momentum
                    history_len = len(self.gradient_history)
                    gradient_weights = np.arange(1, history_len)
                    for t in gradient_weights:
                        self.theta += self.gradient_history[-t] * (1 - b) * b ** (history_len - t)
                # update the gradient
                # num_of_iter += 1

            cur_acc = self.accuracy(x, y)
            if self.verbose:
                print("current accuracy = {}".format(cur_acc))
                print("————————————————————————————————————————————————————————————————————————————————")
            self.accuracy_history.append(cur_acc)

            epoch += 1

        if np.linalg.norm(cur_gradient) <= self.epsilon:
            self.is_converged = True
        if self.verbose:
            print(
                f'terminated after epochs {epoch},  with norm of the gradient equal to {np.linalg.norm(cur_gradient)}')
            print(f'the weight found: {self.theta}')
        return self

    def converged(self):
        return self.is_converged

    def convergence_path(self, index):

        # index : the number of epoch
        # return the accuracy after the training of the index epoch as an array
        return self.accuracy_history[:index]

    def cost_fn(self, x, y, w):
        N, D = x.shape
        z = np.dot(x, w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(
            np.exp(z)))  # log1p calculates log(1+x) to remove floating point inaccuracies
        return J

    def predict(self, x):
        logistic = lambda z: 1. / (1 + np.exp(-z))  # logistic function
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.theta.shape[0] != x.shape[1]:
            x = np.column_stack([x, np.ones(Nt)])
        yh = logistic(np.dot(x, self.theta))  # predict output
        # return yh => [0.98, 0.96, 0.01..]
        res = np.array(yh >= 0.5, dtype='int')
        # print(res)
        return res

    def R_square_score(self, X_test, y_test):
        return R_square(y_test, self.predict(X_test))

    def accuracy(self, X_test, y_test):
        y_hat = self.predict(X_test)

        # print(y_hat)
        return sum(y_hat == y_test) / len(y_hat)
    def description(self):
        # add_bias = True,
        # learning_rate = 1e-4,
        # epsilon = 1e-4,
        # max_epoch = 1000,
        # verbose = False,
        # batch_size = 1,
        # momentum = 0,
        # reset_each_time = True
        print(f"LogisticRegression model with batchingsize = {self.batch_size}, bias = {self.add_bias}, epsilon, = {self.epsilon}"
              f"learning rate = {self.learning_rate}, max_epoch = {self.max_epoch}, momentum = {self.momentum}, reset each time = {self.reset_each_time}"
        )



    def __repr__(self):
        return "LogisticRegression()"
#  Momentnum

class StandardScaler(object):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = [np.mean(X[:, i]) for i in range(X.shape[1])]
        self.scale_ = [np.std(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        # k = X_train.shape
        # for n in k[1] :
        #     X_train[:, n] = (X_train[:, n] - np.mean(X_train[:, n]))/np.std(X_train[:, n])
        return resX

    import numpy as np


