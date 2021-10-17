from utils import *
from LogisticRegression import *

import json
import os.path as osp

def grid_search_LR(train: np.ndarray, val: np.ndarray, param_spaces: List[Dict[str, List[Any]]],
                   measure: Callable[[np.ndarray, np.ndarray], float], verbose: bool = False,
                   force_convergence: bool = False, record_convergence_paths: bool = False
                   ) -> Tuple[Dict[str, Any], float, List[Tuple], List[Tuple]]:
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
    :param force_convergence: If true, then classifiers that fail to converge will not be considered.
    :param record_convergence_paths: The accuracy on the training set after each epoch. If false, the fifth
     return value will be None.
    :return: A five tuple. The first is the best combination of parameters, and the second is its score
    on the validation set. The third is a list of each parameter combination and the corresponding fitted model
    The fourth return value is the convergence path for each combination of parameters.
    '''

    best_score = float('-inf')
    best_params = None
    combinations = []
    for param_space in param_spaces:
        combinations += get_parameter_combinations(param_space)

    train_labels = train[:, -1].astype(int)
    val_labels = val[:, -1].astype(int)

    convergence_paths = [] if record_convergence_paths else None
    models = []
    for combination in tqdm(combinations):
        if verbose:
            print(f"Trying combination {combination}")
        clf = LogisticRegression(**combination)

        clf.fit(train[:, :-1], train_labels)
        if force_convergence and not clf.converged():
            continue
        train_pred, val_pred = clf.predict(train[:, :-1]), clf.predict(val[:, :-1])

        score = measure(val_pred, val_labels)
        models += [(combination, clf)]
        if record_convergence_paths:
            convergence_paths += [(combination, clf.convergence_path())]
        if score > best_score:
            best_score = score
            best_params = combination
        if verbose:
            print(f"score: {score}, \
            {'converged' if clf.converged() else 'not converged, gradient '+str(float(np.linalg.norm(clf.last_gradient)))}")

    return best_params, best_score, models, convergence_paths

def part1(params, checkpoint_step, f_json = None, f_img = None):
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    training = read(training_path, True)
    test = read(test_path)
    validation = read(validation_path)
    pipeline = []
    train_processed, val_processed = preprocess(training, validation, pipeline)

    clf = LogisticRegression(**params)
    clf.fit(training[:, :-1], training[:,-1].astype(int))
    train_pred, val_pred = clf.predict(training[:, :-1]), clf.predict(validation[:, :-1])
    path = clf.convergence_path()

    script_dir = osp.dirname(__file__)
    content_dict = {}
    content_dict["params"] = params
    content_dict["score_on_val"] = accuracy(val_pred, validation[:,-1].astype(int))
    values = []
    checkpoints = np.arange(0, len(path), checkpoint_step)
    if len(path) % checkpoint_step != 0:
        checkpoints = np.append(checkpoints, [len(path)-1])
    for i in checkpoints:
        content_dict[str(i)] = path[i]
        values.append(path[i])

    content = json.dumps(content_dict, indent=2)
    if f_json:
        path = osp.join(script_dir, f_json)
        with open(path, 'w') as f:
            f.write(content)
    plt.clf()
    plt.plot(checkpoints.tolist(), values,
             label=f"{params['max_epoch']} epochs, learning rate {params['learning_rate']}")
    plt.ylim(top=0.5,bottom=0)
    plt.xlabel('Number of epochs')
    plt.ylabel('Norm of gradient')
    plt.title('Convergence path')
    plt.legend()
    plt.show()
    if f_img:
        plt.savefig(f_img, format='png')
    return

def part2(sizes, checkpoint_step, f_json = None, f_img = None):
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    training = read(training_path, True)
    test = read(test_path)
    validation = read(validation_path)
    pipeline = []
    spaces = [{"batch_size": sizes}]
    train_processed, val_processed = preprocess(training, validation, pipeline)
    best_params, best_score, _, _, paths = grid_search_LR(train=train_processed, val=val_processed, param_spaces=spaces,
                                                          measure=accuracy, verbose=True, record_convergence_paths=True)
    path = None
    for combination, p in paths:
        if best_params == combination:
            path = p
    if not path:
        path = paths[0][1]
        print("Error: did not found the best params in paths")
    script_dir = osp.dirname(__file__)
    content_dict = {}
    content_dict["params"] = best_params
    content_dict["score_on_val"] = best_score
    accuracies = []
    checkpoints = np.arange(0, len(path), checkpoint_step)
    if len(path) % checkpoint_step != 0:
        checkpoints = np.append(checkpoints, [len(path) - 1])
    for i in checkpoints:
        content_dict[str(i)] = path[i]
        accuracies.append(path[i])

    content = json.dumps(content_dict, indent=2)
    if f_json:
        path = osp.join(script_dir, f_json)
        with open(path, 'w') as f:
            f.write(content)
    plt.clf()
    plt.plot(checkpoints.tolist(), accuracies,
             label=f"{best_params['max_epoch']} epochs, learning rate {best_params['learning_rate']}")
    plt.ylim(top=0.5, bottom=0)
    plt.xlabel('Number of epochs')
    plt.ylabel('Average accuracy on training set')
    plt.title('Convergence path')
    plt.legend()
    plt.show()
    if f_img:
        plt.savefig(f_img, format='png')
    return

def part3(momentums, f_json = None, f_img = None):
    pass

def part4(small_size, large_size, f_json = None, f_img = None):
    pass


if __name__=='__main__':
    space1 = {'max_epoch': [10000],
              'learning_rate': [0.001],
              'penalty': ['l2', 'l1'],
              'lambdaa': [0.1, 0.01],
              'batch_size': [float('inf'), 16, 256],
              'momentum': [0, 0.99, 0.9]
              }
    space2 = {'max_epoch': [10000],
              'learning_rate': [0.001],
              'penalty': [None],
              'batch_size': [float('inf')],
              'momentum': [0]
              }
    space3 = {'max_epoch': [3000],
              'learning_rate': [0.0002],
              'penalty': [None],
              'batch_size': [float('inf')],
              'momentum': [0]
              }
    params = {'max_epoch': 8000000,
              'learning_rate': 0.0002,
              'penalty': None,
              'batch_size': float('inf'),
              'momentum': 0
              }
    part1(params, 1000, f_json="results/epoch_vs_acc.json", f_img="results/epoch_vs_acc.jpg")

    params = {'max_epoch': 8000000,
              'learning_rate': 0.0001,
              'penalty': None,
              'batch_size': float('inf'),
              'momentum': 0
              }
    part1(params, 1000, f_json="results/epoch_vs_acc2.json", f_img="results/epoch_vs_acc2.jpg")




