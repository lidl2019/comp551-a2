from utils import *
from LogisticRegression import *


def grid_search_LR(train: np.ndarray,
                   val: np.ndarray,
                   param_spaces: List[Dict[str, List[Any]]],
                   measure: Callable[[np.ndarray, np.ndarray], float],
                   report_train: bool = False,
                   report_val: bool = False,
                   report: Callable[[np.ndarray, np.ndarray], Any] = None,
                   verbose: bool = False,
                   force_convergence: bool = False,
                   record_convergence_paths: bool = False,
                   ) -> Tuple[Dict[str, Any], np.ndarray, List[Tuple], List[Tuple], List[Tuple]]:
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
    :param verbose: whether print messages
    :param force_convergence: If true, then classifiers that fail to converge will not be considered.
    :param record_convergence_paths: The accuracy on the training set after each epoch. If false, the fifth
     return value will be None.
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

    train_labels = train[:, -1].astype(int)
    val_labels = val[:, -1].astype(int)

    training_reports = [] if report_train else None
    val_reports = [] if report_val else None
    convergence_paths = [] if record_convergence_paths else None
    for combination in tqdm(combinations):
        if verbose:
            print(f"Trying combination {combination}")
        clf = LogisticRegression(**combination)

        clf.fit(train[:, :-1], train_labels)
        if force_convergence and not clf.converged():
            continue
        train_pred, val_pred = clf.predict(train[:, :-1]), clf.predict(val[:, :-1])

        score = measure(val_pred, val_labels)
        if report_train:
            training_reports += [(combination, report(train_pred, train_labels))]
        if report_val:
            val_reports += [(combination, report(val_pred, val_labels))]
        if record_convergence_paths:
            convergence_paths += [(combination, clf.convergence_path())]
        if score > best_score:
            best_score = score
            best_params = combination
            best_pred = val_pred
        if verbose:
            print(f"score: {score}, \
            {'converged' if clf.converged() else 'not converged, gradient '+str(float(np.linalg.norm(clf.last_gradient)))}")

    return best_params, best_pred, training_reports, val_reports, convergence_paths


if __name__=='__main__':
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    training = read(training_path, True)
    test = read(test_path)
    validation = read(validation_path)

    pipeline = [preprocess_scale]
    space1 = {'max_epoch': [1000],
              'learning_rate': [0.01, 0.001],
              'penalty': ['l2', 'l1'],
              'lambdaa': [0.1, 0.01],
              'batch_size': [float('inf'), 16, 256],
              'momentum': [0, 0.5, 0.9]
              }
    space2 = {'max_epoch': [100000],
              'learning_rate': [0.001],
              'penalty': [None],
              'batch_size': [float('inf')],
              'momentum': [0]
              }
    train_processed, val_processed = preprocess(training, validation, pipeline)
    best_params, best_pred, _, _, paths = grid_search_LR(train=train_processed,
                                                         val=val_processed,
                                                         param_spaces=[space1],
                                                         measure=accuracy,
                                                         record_convergence_paths=True,
                                                         verbose=True)

