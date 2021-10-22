from measures import *
from tqdm import tqdm

class LogisticRegression(object):

    def __init__(self, add_bias=True,
                 learning_rate=1e-4,
                 epsilon=5e-3,
                 max_epoch=100,
                 verbose=False,
                 batch_size=1,
                 momentum=0,
                 reset_each_time = True,
                 penalty=None,
                 lambdaa=0.1,
                 record=False,
                 record_step=1000,
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
        self.reset_each_time = reset_each_time
        self.penalty = penalty
        self.lambdaa = lambdaa
        self.epochs = 1
        self.record = record
        self.record_step = record_step
        self.grad_norm = None
        self.grad_hist = []
        self.acc_hist = []
        self.acc_hist_val = []

    def reset(self):
        self.theta = None
        self.is_converged = False

    def gradient(self, x, y):
        N, D = x.shape
        yh = logistic(np.dot(x, self.theta))  # predictions  size N
        grad = np.dot(x.T, yh - y) / N  # divide by N because cost is mean over N points
        if self.penalty == 'l1':
            grad[:-1] += self.lambdaa * np.sign(self.theta[1:])
        elif self.penalty == 'l2':
            grad[:-1] += self.lambdaa * self.theta[1:]
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
        shuffle = np.random.permutation(x.shape[0])
        new_x = x[shuffle]
        new_y = y[shuffle]
        return new_x, new_y

    def fit(self, x, y, show_progress=False, x_val=None, y_val=None):
        if self.reset_each_time:
            self.reset()
        # mini-batch ->
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])

        N, D = x.shape
        self.theta = np.zeros(D)
        self.last_gradient = np.zeros(D)
        # inital_gradient
        cur_gradient = np.inf

        if self.verbose:
            print("current batch_size = {}".format(self.batch_size))
        self.epochs = 1
        iterable = tqdm(range(1, self.max_epoch+1)) if show_progress else range(1, self.max_epoch+1)
        for epoch in iterable:
            grad_after_epoch = self.gradient(x, y)

            if self.record and epoch % self.record_step == 0:
                self.grad_hist.append(np.linalg.norm(grad_after_epoch))
                self.acc_hist.append(self.accuracy(x, y))
                if not (x_val is None or y_val is None):
                    self.acc_hist_val.append(self.accuracy(x_val, y_val))


            conv_condition = np.linalg.norm(grad_after_epoch) <= self.epsilon
            if conv_condition:
                self.is_converged = True
                break

            if self.batch_size >= N:
                if not self.momentum:
                    self.theta -= self.learning_rate * grad_after_epoch
                else:
                    b = self.momentum
                    cur_gradient = b * self.last_gradient + (1 - b) * grad_after_epoch
                    self.last_gradient = cur_gradient
                    self.theta -= self.learning_rate * cur_gradient
            else:
                new_x, new_y = self.shuffle(x, y)
                # everytime go over the whole dataset, reshuffle the dataset once
                batched_data = self.split_data(new_x, new_y)
                # [(x, y),(x, y),(x, y),(x, y),(x, y)......(x, y)] according to batching
                new_x, new_y = self.shuffle(x, y)
                # everytime go over the whole dataset, reshuffle the dataset once
                batched_data = self.split_data(new_x, new_y)
                # [(x, y),(x, y),(x, y),(x, y),(x, y)......(x, y)] according to batching

                for batched_x, batched_y in batched_data:
                    # go over the whole dataset once according to the batch_size

                    cur_gradient = self.gradient(batched_x, batched_y)
                    if not self.momentum:
                        self.theta -= self.learning_rate * cur_gradient
                    else:
                        b = self.momentum
                        cur_gradient = b * self.last_gradient + (1-b) * cur_gradient
                        self.last_gradient = cur_gradient
                        self.theta -= self.learning_rate * cur_gradient
                    # update the gradient

        self.epochs = epoch
        self.grad_norm = np.linalg.norm(self.gradient(x, y))
        if self.record and epoch % self.record_step != 0:
            self.grad_hist.append(np.linalg.norm(grad_after_epoch))
            self.acc_hist.append(self.accuracy(x, y))
            if not (x_val is None or y_val is None):
                self.acc_hist_val.append(self.accuracy(x_val, y_val))
        if self.verbose:
            print(
                f'terminated after epochs {self.epochs},  with norm of the gradient equal to {np.linalg.norm(cur_gradient)}')
            print(f'the weight found: {self.theta}')
        return self

    def converged(self):
        return self.is_converged

    def cost_fn(self, x, y):
        N, D = x.shape
        z = np.dot(x, self.theta)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(
            np.exp(z)))  # log1p calculates log(1+x) to remove floating point inaccuracies
        return J

    def predict(self, x):
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

    def predict_prob(self,x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.theta.shape[0] != x.shape[1]:
            x = np.column_stack([x, np.ones(Nt)])
        yh = logistic(np.dot(x, self.theta))  # predict output
        return yh

    def R_square_score(self, X_test, y_test):
        return R_square(y_test, self.predict(X_test))

    def accuracy(self, x_test, y_test):
        y_hat = self.predict(x_test)

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

logistic = lambda z: 1. / (1 + np.exp(-z))

