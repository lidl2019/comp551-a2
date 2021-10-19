from utils import *
from LogisticRegression import *

import json
import os.path as osp


def grid_search_LR(train: np.ndarray,
                   val: np.ndarray,
                   param_spaces: List[Dict[str, List[Any]]],
                   measure: Callable[[np.ndarray, np.ndarray], float],
                   verbose: bool = False,
                   force_convergence: bool = False,
                   checkpoint_step: int = -1,
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
    :param force_convergence: If true, then classifiers that fail to converge will not be considered.
    :param checkpoint_step: the checkpoint step for
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
        if checkpoint_step != -1:
            clf.record_step = checkpoint_step
            clf.record = True

        clf.fit(train[:, :-1], train_labels, True, val[:,:-1], val_labels)
        if force_convergence and not clf.converged():
            continue
        train_pred, val_pred = clf.predict(train[:, :-1]), clf.predict(val[:, :-1])

        score = measure(val_pred, val_labels)
        models += [(combination, clf)]
        if score > best_score:
            best_score = score
            best_params = combination
        if verbose:
            print(f"score: {score}, \
            {'converged' if clf.converged() else 'not converged, gradient '+str(float(np.linalg.norm(clf.last_gradient)))}")

    return best_params, best_score, models

def q1(params, checkpoint_step, f_img1 = None, f_img2 = None):
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    training = read(training_path, True)
    test = read(test_path)
    validation = read(validation_path)
    pipeline = []
    training, validation = preprocess(training, validation, pipeline)
    params["record_step"] = checkpoint_step
    clf = LogisticRegression(**params)
    clf.fit(training[:, :-1], training[:,-1], True, x_val=validation[:, :-1], y_val=validation[:,-1])
    train_pred, val_pred = clf.predict(training[:, :-1]), clf.predict(validation[:, :-1])

    print(accuracy(train_pred, training[:, -1]))
    print(accuracy(val_pred, validation[:,-1]))
    script_dir = osp.dirname(__file__)
    content_dict = {}
    content_dict["params"] = params
    content_dict["score_on_val"] = accuracy(val_pred, validation[:,-1].astype(int))
    acc = clf.acc_hist
    val_acc = clf.acc_hist_val
    epochs = clf.epochs
    checkpoints = checkpoint_step * np.arange(0, len(acc))
    plt.clf()
    plt.plot(checkpoints.tolist(), acc,
             label="training", alpha=0.5)
    plt.plot(checkpoints.tolist(), val_acc,
             label="validation", alpha=0.5)
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracies for {epochs} epochs, learning rate {params['learning_rate']}")
    plt.legend()
    if f_img1:
        plt.savefig(f_img1)
    plt.show()
    plt.close()

    grad = clf.grad_hist
    plt.plot(checkpoints.tolist(), grad)
    #plt.ylim(top=.05,bottom=0)
    plt.xlabel('Number of epochs')
    plt.ylabel('Norm of the gradient of the cost')
    plt.title(f"Convergence plot for {epochs} epochs, learning rate {params['learning_rate']}")
    #plt.legend()
    if f_img2:
        plt.savefig(f_img2)
    plt.show()
    plt.close()
    return clf


def q2(sizes: List[int],
       params: Dict[str, Any],
       checkpoint_step: int,
       fnames_acc: List[str]=None,
       fnames_grad: List[str]=None,
       fname_performance: str=None,
       verbose=True):
    if not fnames_acc is None and len(fnames_acc) != len(sizes):
        print("Error: file name list for accuracy does not match sizes in length")
        return
    if not fnames_grad is None and len(fnames_grad) != len(sizes):
        print("Error: file name list for norm of gradient does not match sizes in length")
        return
    training_path = "./data_A2/diabetes/diabetes_train.csv"
    test_path = "./data_A2/diabetes/diabetes_test.csv"
    validation_path = "./data_A2/diabetes/diabetes_val.csv"
    training = read(training_path, True)
    test = read(test_path)
    validation = read(validation_path)
    pipeline = []
    p = dict(params)
    for k, v in p.items():
        p[k] = [v]
    p["batch_size"] = sizes
    spaces = [p]
    train_processed, val_processed = preprocess(training, validation, pipeline)
    best_params, best_score, models = grid_search_LR(train=train_processed,
                                                     val=val_processed,
                                                     param_spaces=spaces,
                                                     measure=accuracy,
                                                     verbose=True)

    content = {}
    for i, (_, model) in enumerate(models):
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
        plt.title(f"Accuracies for {epochs} epochs, batch_size {model.batch_size}")
        plt.legend()
        if fnames_acc:
            plt.savefig(fnames_acc[i])
        if verbose:
            plt.show()
        plt.close()

        grad = model.grad_hist
        plt.plot(checkpoints.tolist(), grad, label="grad")
        plt.xlabel('Number of epochs')
        plt.ylabel('Norm of the gradient of the cost')
        plt.title(f"Convergence plot for {epochs} epochs, batch_size {model.batch_size}")
        if fnames_grad:
            print("save!!!!!")
            plt.savefig(fnames_grad[i])
        if verbose:
            plt.show()
        plt.close()

        if fname_performance:
            content[f"model {i}"] = {
                "batch_size": model.batch_size,
                "lr": model.learning_rate,
                "epochs": model.epochs,
                "training accuracy": acc[-1],
                "validation accuracy": val_acc[-1]
            }
            js = json.dumps(content, indent=2)
            with open(fname_performance, 'w') as f:
                f.write(js)
        if verbose:
            print(f"model with batch size {model.batch_size}:")
            print(f"\ttraining accuracy: {acc[-1]}")
            print(f"\tvalidation accuracy: {val_acc[-1]}")
    return models


def part3(momentums, f_json = None, f_img = None):
    pass


def part4(small_size, large_size, f_json = None, f_img = None):
    pass


if __name__=='__main__':
    # region part1
    params = {'max_epoch': 1,
              'learning_rate': 0.0003,
              'batch_size': float('inf'),
              'momentum': 0,
              'record': True,
              }
    im1 = "results/1.1/epoch_vs_acc.jpg"
    im2 = "results/1.1/epoch_vs_grad.jpg"

    clf = q1(params, 101, f_img1=None, f_img2=None)
    # endregion

    params = {'max_epoch': 100000,
              'learning_rate': 0.0675,
              'batch_size': float('inf'),
              'momentum': 0.99,
              'record': True,
              }
    sizes = [600, 601, 602, 603]
    acc_names = [f"results/1.2/acc-{i}.jpg" for i in sizes]
    grad_names = [f"results/1.2/norm-{i}.jpg" for i in sizes]
    perf_name = "results/1.2/performance.json"
    models = q2(sizes, params, 1001, acc_names, grad_names, perf_name)