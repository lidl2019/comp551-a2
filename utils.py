from typing import List, Tuple, Callable, Dict, Any, Union
import pandas as pd
import numpy as np

class LogisticRegression:
    pass

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
            resX[:, col] = (X[:, col] - self.mean_[col])/self.scale_[col]

        # k = X_train.shape
        # for n in k[1] :
        #     X_train[:, n] = (X_train[:, n] - np.mean(X_train[:, n]))/np.std(X_train[:, n])
        return resX

def accuracy_score(y_pred, y_true):
    return sum(y_pred == y_true)/len(y_true)
