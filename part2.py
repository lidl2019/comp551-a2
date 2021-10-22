from utils import *
from measures import *

from sklearn.linear_model import LogisticRegression as skLR
from sklearn.linear_model import SGDClassifier
from LogisticRegression import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

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

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
def preprocess_LSA(train: np.ndarray,
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
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    x = train[:, :-1]
    y = train[:, -1, np.newaxis]
    x = csr_matrix(x)
    x = svd.fit_transform(x)
    train = np.append(x, y, axis=1).astype(float)

    x = test[:, :-1]
    y = test[:, -1, np.newaxis]
    x = csr_matrix(x)
    x = svd.transform(x)
    test = np.append(x, y, axis=1).astype(float)
    return train, test

def preprocess_check_occurrence(train: np.ndarray,
                                test: np.ndarray,
                                vectorizer: Vectorizer = None,
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    turn count features into occurrence features
    :param train:
    :param test:
    :param vectorizer:
    :return:
    """
    x = np.array(train[:, :-1] > 0, dtype=int)
    y = train[:, -1, np.newaxis]
    train = np.append(x, y, axis=1).astype(float)

    x = np.array(test[:, :-1] > 0, dtype=int)
    y = test[:, -1, np.newaxis]
    test = np.append(x, y, axis=1).astype(float)
    return train, test

def grid_search_LR(train: np.ndarray,
                   val: np.ndarray,
                   param_spaces: List[Dict[str, List[Any]]],
                   measure: Callable[[np.ndarray, np.ndarray], float],
                   verbose: bool = False,
                   ) -> Tuple[Dict[str, Any], float, List[Tuple]]:
    '''
    Finds the best choice of hyperparameters, given the range of values of each parameter. Optionally returns the
    performance report of all parameter combinations on training or validation set.
    :param train: the training set
    :param val: the validation set
    :param param_spaces: a dictionary containing parameter names as keys, and ranges of parameter values as values
    :param measure: the measure to optimize for. The higher the measure, the better the model.
    the third return value will be None.
    the fourth return value will be None.
    predictions and labels as input
    :param verbose: whether print messages
    :return: A three tuple. The first is the best combination of parameters, and the second is its score
    on the validation set. The third is a list of each parameter combination and the corresponding fitted model
    '''

    best_score = float('-inf')
    best_params = None
    combinations = []
    for param_space in param_spaces:
        combinations += get_parameter_combinations(param_space)

    train_labels = train[:, -1].astype(int)
    val_labels = val[:, -1].astype(int)

    models = []
    for combination in tqdm(combinations):
        if verbose:
            print(f"Trying combination {combination}")
        clf = LogisticRegression(**combination)

        clf.fit(train[:, :-1], train_labels)

        train_pred, val_pred = clf.predict(train[:, :-1]), clf.predict(val[:, :-1])

        score = measure(val_pred, val_labels)
        models += [(combination, clf)]
        if score > best_score:
            best_score = score
            best_params = combination
        if verbose:
            print(f"score: {score}")

    return best_params, best_score, models

def try_pipeline(pipeline, dir, param):
    model = LogisticRegression(**param)
    model.record = True
    train, val = read_processed_data(pipeline, dir)
    model.fit(train[:, :-1], train[:, -1], True, val[:, :-1], val[:, -1])
    acc = model.acc_hist
    val_acc = model.acc_hist_val
    checkpoints = checkpoint_step * np.arange(0, len(acc))
    epochs = model.epochs
    plt.plot(checkpoints.tolist(), acc,
             label="training", alpha=0.5)
    plt.plot(checkpoints.tolist(), val_acc,
             label="validation", alpha=0.5)
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracies for {epochs} epochs")
    plt.legend()
    plt.show()
    plt.close()
    print(val_acc[-1])

    grad = model.grad_hist
    plt.plot(checkpoints.tolist(), grad, label="grad")
    plt.xlabel('Number of epochs')
    plt.ylabel('Norm of the gradient of the cost')
    plt.title(f"Convergence plot for {epochs} epochs")
    plt.show()
    plt.close()
    return model

from sklearn.linear_model import LogisticRegression as skLR

if __name__=='__main__':
    training_path = "data_A2/fake_news/fake_news_train.csv"
    test_path = "data_A2/fake_news/fake_news_test.csv"
    validation_path = "data_A2/fake_news/fake_news_val.csv"
    training = read(training_path)
    validation = read(validation_path)
    all_pipelines = {
        '20k-count-(1,1)-scaled': ([
                                       preprocess_vectorize, preprocess_scale
                                   ], build_vectorizer(max_features=20000, ngram_range=(1, 2))),
        '20k-count-(1,2)-scaled': ([
                                    preprocess_vectorize, preprocess_scale
                                        ], build_vectorizer(max_features=20000, ngram_range=(1, 2))),
        '20k-tfidf-(1,1)-scaled': ([
                                       preprocess_vectorize, preprocess_scale
                                   ], build_vectorizer(max_features=20000, type='tfidf', ngram_range=(1, 2))),
        '20k-tfidf-(1,2)-scaled': ([
                                       preprocess_vectorize, preprocess_scale
                                   ], build_vectorizer(max_features=20000, type='tfidf', ngram_range=(1, 2))),
        '20k-tfidf-stem-(1,2)-scaled': ([
                   preprocess_vectorize, preprocess_scale
               ], build_vectorizer(max_features=20000, type='tfidf', normalization='stemming', ngram_range=(1, 2))),
        '20k-tfidf-stem-(1,3)-scaled': ([
                   preprocess_vectorize, preprocess_scale
               ], build_vectorizer(max_features=20000, type='tfidf', normalization='stemming', ngram_range=(1, 3))),
        '20k-tfidf-stem-scaled': ([
                                      preprocess_vectorize, preprocess_scale
                                  ], build_vectorizer(max_features=20000, type='tfidf', normalization='stemming')),
        '20k-tfidf-stopwords': ([
                                    preprocess_vectorize, preprocess_scale
                                ], build_vectorizer(max_features=20000, type='tfidf',
                                                    stopwords=stopwords.words("english"))),
    }

    pipeline_base = {
    }

    #save_processed_data(training, validation, all_pipelines, 'p2_processed')
    checkpoint_step = 15
    param = {'max_epoch': 300,
             'learning_rate': 0.1,
             'penalty': None,
             'batch_size': float('inf'),
             'momentum': 0.99,
             'record': True,
             'record_step': checkpoint_step
             }
    #model = try_pipeline('20k-tfidf-(1,2)-scaled', 'p2_processed', param)

    grid = {
        'loss': ['log'],
        'max_iter': [100],
        'penalty': ['elasticnet'],
        'l1_ratio': [0, 0.15, 1],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': [0.00001, 0.0001, 0.01, 0.1],
        'early_stopping': [True, False]
    }

    train, val = read_processed_data('20k-count-(1,2)-scaled', 'p2_processed')
    print('starting grid search')
    cv = GridSearchCV(SGDClassifier(), grid, 'accuracy')
    cv.fit(train[:,:-1], train[:,-1].astype(int))
    model = cv.best_estimator_
    print(cv.best_params_)
    print(cv.best_score_)

    space = {'max_epoch': [200],
             'learning_rate': [0.1],
             'penalty': [None],
             'batch_size': [float('inf')],
             'momentum': [0.99],
             'record': [False],
             'record_step': [1000]
             }
    '''best_pipeline, best_score, searched_params, _pipeline_scores = grid_search_processed_data(all_pipelines,
                                                                                              'p2_processed',
                                                                                              [space],
                                                                                              accuracy,
                                                                                              grid_search_LR,
                                                                                              verbose=True)'''


