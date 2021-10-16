import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def accuracy(y_pred, y_true):
    return sum(y_pred == y_true)/len(y_true)


def get_f1(average="macro"):
    return lambda y_pred, y_true: f1_score(y_pred=y_pred, y_true=y_true, average=average)


def mean_square_error(y_pred, y_true):
    assert len(y_true) == len(y_pred)
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def mae(y_pred, y_true):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def R_square(y_pred, y_true):
    return 1 - mean_square_error(y_true, y_pred) / np.var(y_true)


def get_reporter(kwargs=dict()):
    return lambda y_pred, y_true: classification_report(y_pred=y_pred, y_true=y_true, **kwargs, output_dict=True)
