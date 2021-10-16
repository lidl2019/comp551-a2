from utils import *
from measures import *

#from sklearn.linear_model import LogisticRegression
from LogisticRegression import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


# region preprocessing


def build_vectorizer(type: str = "count",
                     max_features: int = None,
                     ngram_range: Tuple[int, int] = (1,1),
                     lowercase: bool = True,
                     normalization: str = None,
                     stopwords: Iterable[str] = ()
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
                         vectorizer: Vectorizer = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the data by the given vectorizer
    :param train: the training set to be transformed
    :param test: the test set to be transformed
    :param vectorizer: the vectorizer
    :return: transformed data
    """
    if not vectorizer:
        vectorizer = build_vectorizer()
    x = train[:, 0]
    y = train[:, -1, np.newaxis]
    x = vectorizer.fit_transform(x).toarray()
    train = np.append(x, y, axis=1).astype(float)

    x = test[:, 0]
    y = test[:, -1, np.newaxis]
    x = vectorizer.transform(x).toarray()
    test = np.append(x, y, axis=1).astype(float)
    return train, test


def grid_search_LR(train: np.ndarray,
                   val: np.ndarray,
                   param_spaces: List[Dict[str, List[Any]]],
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
        :param param_spaces: a dictionary containing parameter names as keys, and ranges of parameter values as values
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
    combinations = []
    for param_space in param_spaces:
        combinations += get_parameter_combinations(param_space)

    train_labels = train[:,-1].astype(int)
    val_labels = val[:,-1].astype(int)

    training_reports = []
    val_reports = []
    for combination in tqdm(combinations):
        if verbose:
            print(f"Trying combination {combination}")
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
        if verbose:
            print(f"score: {score}, \
            {'converged' if clf.converged() else 'not converged, gradient ' + str(float(np.linalg.norm(clf.last_gradient)))}")

    return best_params, best_pred, training_reports, val_reports


if __name__=='__main__':
    training_path = "data_A2/fake_news/fake_news_train.csv"
    test_path = "data_A2/fake_news/fake_news_test.csv"
    validation_path = "data_A2/fake_news/fake_news_val.csv"
    training = read(training_path)
    validation = read(validation_path)

    space1 = {'max_epoch': [500],
              'learning_rate': [0.05, 0.001],
              'penalty': ['l2', 'l1'],
              'lambdaa': [0.1, 0.01],
              'batch_size': [float('inf'), 1024],
              'momentum': [0, 0.9]
              }
    space2 = {'max_epoch': [500],
              'learning_rate': [0.05, 0.001],
              'penalty': [None],
              'batch_size': [float('inf'), 1024],
              'momentum': [0, 0.9]
              }

    pipelines = {

        'p1': ([
                   preprocess_vectorize,
               ], build_vectorizer(max_features=10000, type='tfidf', normalization='lemmatization', ngram_range=(1, 2))),
        'p2': ([
                   preprocess_vectorize,
               ], build_vectorizer(max_features=10000, type='tfidf', normalization='stemming', ngram_range=(1, 2))),
        'p3': ([
                   preprocess_vectorize,
               ], build_vectorizer(max_features=10000, type='tfidf', normalization='stemming', ngram_range=(1, 3))),
    }
    best_pipeline, best_score, searched_params, _ = grid_search_pipelines(train=training,
                                                                          val=validation,
                                                                          param_spaces=[space1, space2],
                                                                          pipelines=pipelines,
                                                                          measure=accuracy,
                                                                          search_method=grid_search_LR)


