import numpy as np
def shuffle(all_data, all_labels):
    data = np.stack((all_data, all_labels), axis=1)
# data.shape
    np.random.shuffle(data)
    return data
def accuracy_score(y_hat, y_true):
    return sum(y_hat == y_true) / len(y_true)


def mean_square_error(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def mae(y_true, y_predict):
    return np.sum(np.abs(y_true - y_predict)) / len(y_true)


def R_square(y_true, y_predict):
    return 1 - mean_square_error(y_true, y_predict) / np.var(y_true)


def rse(y_true, y_predict):
    return np.sqrt(mean_square_error(y_true, y_predict))