import pandas as pd
import numpy as np
from utils import get_parameter_combinations, read, accuracy_score, fit_scaler, scale

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
from tqdm import tqdm

Vectorizer = Union[TfidfVectorizer, CountVectorizer]


# region preprocessing


def build_vectorizer(type: str = "count",
                     max_features: int = None,
                     ngram_range: Tuple[int, int] = (1,1),
                     lowercase: bool = True,
                     normalization: str = None,
                     stopwords: Iterable = ()
                     ) -> Vectorizer:
    '''
    Build a vectorizer.
    :param type: {"count", "tf", "tfidf"}, default = "count" the type of features to extract
    :param max_features: the maximum number of features
    :param ngram_range: (min_ngram, max_ngram), the range of ngrams
    :param lowercase: whether to set lowercase before tokenization
    :param normalization: {None, "stemming", "lemmatization"} the choice of normalization technique
    :param stopwords: the collection of stopwords to be ignored when tokenizing
    :return: the vectorizer specified.
    '''

    default_tokenizer = CountVectorizer().build_tokenizer()
    if normalization == 'stemming':
        stemmer = SnowballStemmer("english")
        tokenizer = lambda s: [stemmer.stem(w) for w in default_tokenizer(s) if w not in stopwords]
    elif normalization == 'lemmatization':
        wnl = WordNetLemmatizer()
        tokenizer = lambda s: [wnl.lemmatize(w) for w in default_tokenizer(s) if w not in stopwords]
    else:
        tokenizer = lambda s: [w for w in default_tokenizer(s) if w not in stopwords]

    if type == "tf":
        vectorizer = TfidfVectorizer(use_idf=False,
                                     max_features=max_features,
                                     ngram_range=ngram_range,
                                     lowercase=lowercase,
                                     tokenizer=tokenizer)
    elif type =="tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features,
                                     ngram_range=ngram_range,
                                     lowercase=lowercase,
                                     tokenizer=tokenizer)
    else:
        vectorizer = CountVectorizer(max_features=max_features,
                                     ngram_range=ngram_range,
                                     lowercase=lowercase,
                                     tokenizer=tokenizer)
    return vectorizer


def preprocess_vectorize(train: np.ndarray,
                         test: np.ndarray,
                         vectorizer: Vectorizer
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the data by the given vectorizer
    :param train: the training set to be transformed
    :param test: the test set to be transformed
    :param vectorizer: the vectorizer
    :return: transformed data
    """
    x = train[:, 0]
    y = train[:, -1, np.newaxis]
    x = vectorizer.fit_transform(x).toarray()
    train = np.append(x, y, axis=1)

    x = test[:, 0]
    y = test[:, -1, np.newaxis]
    x = vectorizer.transform(x).toarray()
    test = np.append(x, y, axis=1)
    return train, test


def preprocess_scale(train: np.ndarray,
                     test: np.ndarray,
                     vectorizer: Vectorizer
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


Pipeline = List[Callable[[np.ndarray, np.ndarray, Vectorizer], Tuple[np.ndarray, np.ndarray]]]


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


def grid_search_LR(train: np.ndarray,
                   val: np.ndarray,
                   ranges: Dict[str, List[Any]],
                   measure: Callable[[np.ndarray, np.ndarray], float],
                   report_train: bool = False,
                   report_val: bool = False,
                   report: Callable[[np.ndarray, np.ndarray], Any] = None,
                   verbose: bool = False
                   ) -> Tuple[Dict[str, Any], np.ndarray, List[Any], List[Any]]:
    '''
        Finds the best choice of hyperparameters, given the range of values of each parameter. Optionally returns the
        performance report of all parameter combinations on training or validation set.
        :param train: the training set
        :param val: the validation set
        :param ranges: a dictionary containing parameter names as keys, and ranges of parameter values as values
        :param measure: the measure to optimize for. The higher the measure, the better the model.
        :param report_train: whether to report the performance of each combination on the training set. If false,
        the third return value will be None.
        :param report_val: whether to report the performance of each combination on the validation set. If false,
        the fourth return value will be None.
        :param report: the report function used for training and validation sets. The report function should take
        predictions and labels as input
        :param force_convergence: If true, then classifiers that fail to converge will not be considered. If false,
        the fifth return value will be None.
        :param record_convergence_paths: If true, then the fifth return value will be non-empty
        :param verbose: whether print messages
        :return: A five tuple. The first is the best combination of parameters, and the second is its predictions
        on the validation set. The third is a list of each parameter combination and its report on the training set.
        The fourth return value is the counterpart of the third for validation set.
        The fifth return value is the convergence path for each combination of parameters.
        '''
    best_score = float('-inf')
    best_params = None
    best_pred = None
    combinations = get_parameter_combinations(ranges)

    train_labels = train[:,-1].astype(int)
    val_labels = val[:,-1].astype(int)

    training_reports = []
    val_reports = []
    for combination in tqdm(combinations):
        clf = LogisticRegression(**combination)

        clf.fit(train[:, :-1], train_labels)
        train_pred, val_pred = clf.predict(train[:, :-1]), clf.predict(val[:, :-1])

        score = measure(val_pred, val_labels)
        if report_train:
            training_reports += (combination, report(train_pred, train_labels))
        if report_val:
            val_reports += (combination, report(val_pred, val_labels))
        if score > best_score:
            best_score = score
            best_params = combination
            best_pred = val_pred

    return best_params, best_pred, training_reports, val_reports


def grid_search_pipelines(train: np.ndarray,
                          val: np.ndarray,
                          ranges: Dict[str, List[Any]],
                          pipelines: Dict[str, Tuple[Pipeline, Vectorizer]],
                          measure: Callable[[np.ndarray, np.ndarray], float],
                          report: Callable[[np.ndarray, np.ndarray], Any] = None,
                          verbose: bool = False
                          ) -> Tuple[Optional[str], float, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compare preprocessing pipelines using grid search
    :param train: the training set
    :param val: the validation set
    :param ranges: the ranges of hyperparameters for the logistic regression model
    :param pipelines: pipelines to be compared. Each come with a name and a specified vectorizer
    :param measure: the measure used for comparing. The higher, the better.
    :param report: optional callable used to report the best performance for each pipeline. If not specified, the fourth
    return value will be None
    :param verbose: whether to print messages
    :return: A four tuple. The first is the name of the best pipeline, and the second is its score. The third is the
    best hyperparameters for each pipeline. The fourth is the best performance report for each pipeline
    """
    best_score = float('-inf')
    best_pipeline = None
    searched_params = {}
    pipeline_reports = None if report else {}
    for name, (pipeline, vectorizer) in pipelines.items():
        train_processed, val_processed = preprocess(train, val, pipeline, vectorizer)
        best_params, val_pred, _, _ = grid_search_LR(train_processed,
                                                     val_processed,
                                                     ranges,
                                                     measure)
        score = measure(val_pred, val_processed[:, -1])
        if report:
            pipeline_reports[name] = report(val_pred, val_processed[:, -1])
        if score > best_score:
            best_score = score
            best_pipeline = name

        searched_params[name] = best_params
    return best_pipeline, best_score, searched_params, pipeline_reports

if __name__=='__main__':
    training_path = "data_A2/fake_news/fake_news_train.csv"
    test_path = "data_A2/fake_news/fake_news_test.csv"
    validation_path = "data_A2/fake_news/fake_news_val.csv"
    training = read(training_path)
    validation = read(validation_path)
    vec = build_vectorizer(max_features=6000)
    params_ranges = {'max_iter': [100, 200]}
    pipelines = {
        'p1': ([
            preprocess_vectorize,
            preprocess_scale
               ], vec),
        'p2': ([
            preprocess_vectorize,
               ], vec),
    }
    best_pipeline, best_score, searched_params, _ = grid_search_pipelines(training,
                                                                          validation,
                                                                          params_ranges,
                                                                          pipelines,
                                                                          accuracy_score)


