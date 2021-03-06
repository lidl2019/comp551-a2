import pandas as pd
import numpy as np
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
import matplotlib.pyplot as plt
from tqdm import tqdm

Vectorizer = Union[TfidfVectorizer, CountVectorizer]
Pipeline = List[Callable[[np.ndarray, np.ndarray, Optional[Vectorizer]], Tuple[np.ndarray, np.ndarray]]]


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


def get_parameter_combinations(ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    '''
    Computes all combinations of parameters given a range for each parameter
    :param ranges: a dictionary containing the range of values for each parameter
    :return: a list of all combinations of the parameter values
    '''
    keys = list(ranges.keys())
    values = list(ranges.values())

    # the cartesian product of the values of different parameters
    value_combinations = list(product(*values))

    # generate all combinations of parameters
    return [{keys[i]: combination[i] for i in range(len(keys))}
            for combination in value_combinations]


def read(path: str, verbose: bool = False) -> np.ndarray:
    '''
    Read the dataset
    :param path: the dataset to be preprocessed
    :param scaler: the fitted standard scaler
    :param verbose: whether print a data preview
    :return: the dataset with features preprocessed
    '''
    data_df = pd.read_csv(path)
    if verbose:
        print(data_df)
    return data_df.to_numpy()


def fit_scaler(data: np.ndarray, verbose: bool = False) -> StandardScaler:
    '''
    Returns a scaler fit on the given dataset
    :param data: the target dataset to be fitted by the scaler
    :param verbose: whether print messages
    :return: the scaler with mean and variance from the target dataset
    '''
    std_scaler = StandardScaler()
    features = data[:,:-1]
    std_scaler.fit(features)
    return std_scaler


def scale(data: np.ndarray, scaler: StandardScaler, verbose: bool = False) -> np.ndarray:
    '''
    Preprocess the data by standard scaling
    :param data: the dataset to be preprocessed
    :param scaler: the fitted standard scaler
    :param verbose: whether print messages
    :return: the dataset with features preprocessed
    '''
    features = data[:, :-1]
    labels = data[:, -1, np.newaxis]
    scaled_features = scaler.transform(features)
    return np.append(scaled_features,labels, axis=1)


def preprocess_scale(train: np.ndarray,
                     test: np.ndarray,
                     vectorizer: Vectorizer = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale the data using standard scaling. The vectorizer parameter is ignored
    :param train: the training set to be scaled
    :param test: the test set to be scaled
    :param vectorizer: ignored
    :return: scaled data
    """
    scaler = fit_scaler(train)
    train = scale(train, scaler)
    test = scale(test, scaler)
    return train, test


def preprocess(train: np.ndarray,
               test: np.ndarray,
               pipeline: Pipeline,
               vectorizer: Vectorizer = None,
               ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Preprocess the dataset given a pipeline of preprocessing
    :param train: the training set
    :param test: the test or validation set
    :param pipeline: the steps of preprocessing to be applied sequentially
    :param vectorizer: the vectorizer that turns text into feature vectors
    :return: the data with preprocessed features and labels
    '''
    for step in pipeline:
        train, test = step(train, test, vectorizer)
    return train, test

# endregion


def grid_search_pipelines(train: np.ndarray,
                          val: np.ndarray,
                          param_spaces: List[Dict[str, List[Any]]],
                          pipelines: Dict[str, Tuple[Pipeline, Vectorizer]],
                          measure: Callable[[np.ndarray, np.ndarray], float],
                          search_method = Any,
                          verbose: bool = False
                          ) -> Tuple[Optional[str], float, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compare preprocessing pipelines using grid search
    :param train: the training set
    :param val: the validation set
    :param param_spaces: the ranges of hyperparameters for the logistic regression model
    :param pipelines: pipelines to be compared. Each come with a name and a specified vectorizer
    :param measure: the measure used for comparing. The higher, the better.
    :param search_method: the function used to search over parameter ranges
    :param verbose: whether to print messages
    :return: A four tuple. The first is the name of the best pipeline, and the second is its score. The third is the
    best hyperparameters for each pipeline. The fourth is the best score for each pipeline
    """
    best_score = float('-inf')
    best_pipeline = None
    searched_params = {}
    pipeline_scores = {}
    for name, (pipeline, vectorizer) in pipelines.items():
        train_processed, val_processed = preprocess(train, val, pipeline, vectorizer)
        best_params, score = search_method(train_processed,
                                                    val_processed,
                                                    param_spaces,
                                                    measure)[:2]
        if score > best_score:
            best_score = score
            best_pipeline = name
        if verbose:
            print(f"Pipeline: {name}, score: {score}, params: {best_params}")
        pipeline_scores[name] = score
        searched_params[name] = best_params

    return best_pipeline, best_score, searched_params, pipeline_scores


def save_processed_data(train: np.ndarray,
                        val: np.ndarray,
                        pipelines: Dict[str, Tuple[Pipeline, Vectorizer]],
                        dir: str) -> None:
    """
    Save processed training and validation sets by different pipelines
    :param train: the training set
    :param val: the validation set
    :param pipelines: the preprocessing pipelines
    :param dir: directory name to store processed training and validation sets
    :return: None
    """
    for name, (pipeline, vectorizer) in pipelines.items():
        train_processed, val_processed = preprocess(train, val, pipeline, vectorizer)
        np.savez_compressed(dir + '/' + name, train = train_processed, val = val_processed)
    return


def read_processed_data(pipeline_name: str, dir: str) -> Tuple[np.array, np.array]:
    """
    Load preprocessed training and validation sets
    :param pipeline_name: the original name of the pipeline
    :param dir: the directory name of processed training and validation sets
    :return:
    """
    loaded = np.load(dir + '/' + pipeline_name + ".npz")
    return loaded["train"], loaded["val"]


def grid_search_processed_data(pipelines: Dict[str, Tuple[Pipeline, Vectorizer]],
                               dir: str,
                               param_spaces: List[Dict[str, List[Any]]],
                               measure: Callable[[np.ndarray, np.ndarray], float],
                               search_method = Any,
                               verbose: bool = False
                               ) -> Tuple[Optional[str], float, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compare preprocessing pipelines using grid search
    :param pipelines: the original pipelines used to store files
    :param dir: the directory name of processed training and validation sets
    :param param_spaces: the ranges of hyperparameters for the logistic regression model
    :param measure: the measure used for comparing. The higher, the better.
    :param search_method: the function used to search over parameter ranges
    :param verbose: whether to print messages
    :return: A four tuple. The first is the name of the best pipeline, and the second is its score. The third is the
    best hyperparameters for each pipeline. The fourth is the best score for each pipeline
    """
    best_score = float('-inf')
    best_pipeline = None
    searched_params = {}
    pipeline_scores = {}
    for name in pipelines:
        train_processed, val_processed = read_processed_data(name, dir)
        best_params, score, models = search_method(train_processed, val_processed, param_spaces, measure)
        if score > best_score:
            best_score = score
            best_pipeline = name
        if verbose:
            print(f"Pipeline: {name}, score: {score}, params: {best_params}")
        pipeline_scores[name] = score
        searched_params[name] = best_params

    return best_pipeline, best_score, searched_params, pipeline_scores

