from utils import *
from itertools import product

def fit_scaler(path: str) -> StandardScaler:
    '''
    Returns a scaler fit on the given dataset
    :param path: the target dataset to be fitted by the scaler
    :return: the scaler with mean and variance from the target dataset
    '''
    std_scaler = StandardScaler()
    features = pd.read_csv(path).iloc[:, :-1].to_numpy()
    std_scaler.fit(features)
    return std_scaler

def preprocess_scaling(path: str, scaler: StandardScaler) -> np.ndarray:
    '''
    Preprocess the dataset by standard scaling
    :param path: the dataset to be preprocessed
    :param scaler: the fitted standard scaler
    :return: the dataset with features preprocessed
    '''
    features = pd.read_csv(path).iloc[:, :-1].to_numpy()
    labels = pd.read_csv(path).iloc[:, -1].to_numpy()[:,np.newaxis]
    scaled_features = scaler.transform(features)
    return np.append(scaled_features,labels, axis=1)


def validate_model(clf: LogisticRegression,
                   training: np.ndarray,
                   validation: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fits the model on training set, and returns predictions on training and validation sets
    :param clf: the classifier
    :param training: training set
    :param validation: validation set
    :return: the prediction result for training and validation sets
    '''
    clf.fit(training[:,:-1], training[:,-1])
    return clf.predict(training[:,:-1]), clf.predict(validation[:,:-1])

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

def grid_search_LR(training: np.ndarray,
                validation: np.ndarray,
                ranges: Dict[str, List[Any]],
                measure: Callable[[np.ndarray, np.ndarray], float],
                report_train : Callable[[np.ndarray, np.ndarray], Any] = None,
                report_val: Callable[[np.ndarray, np.ndarray], Any] = None
                ) -> Tuple(Dict[str, Any], float, List[Any], List[Any]):
    '''
    Finds the best choice of hyperparameters, given the range of values of each parameter. Optionally returns the
    performance report of all parameter combinations on training or validation set.
    :param training: the training set
    :param validation: the validation set
    :param ranges: a dictionary containing parameter names as keys, and ranges of parameter values as values
    :param measure: the measure to optimize for. The higher the measure, the better the model.
    :param report_train: the report function used for the training set. The report function should take predictions
    and labels as input
    :param report_val: the report function used for the validation set. The report function should take predictions
    and labels as input
    :return: A tuple of four values. The first is the best combination of parameters, and the second is its score
    on the validation set. The third is a list of each parameter combination and its report on the training set.
    The fourth return value is the counterpart of the third for validation set.
    '''

    best_score = float('-inf')
    best_params = None
    combinations = get_parameter_combinations(ranges)

    train_labels = training[:][-1]
    val_labels = validation[:][-1]

    training_reports = []
    val_reports = []
    for combination in combinations:
        clf = LogisticRegression(**combination)
        train_pred, val_pred = validate_model(clf, training, validation)
        score = measure(val_pred, val_labels)
        if report_train != None:
            training_reports += (combination, report_train(train_pred, train_labels))
        if report_val != None:
            val_reports += (combination, report_val(val_pred, val_labels))
        if score > best_score:
            best_score = score
            best_params = combination

    return best_params, best_score, training_reports, val_reports


if __name__=='__main__':
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    scaler = fit_scaler(training_path)

    training = preprocess_scaling(training_path, scaler)
    test = preprocess_scaling(test_path, scaler)
    validation = preprocess_scaling(validation_path, scaler)
