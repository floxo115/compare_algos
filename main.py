import configparser
import json
import pathlib
import re
import sys
from collections import Counter
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

config_parser = configparser.ConfigParser()
config_parser.read("config.ini", encoding="utf-8")
SEED = config_parser.getint("testing", "seed")
N_PROCESSES = config_parser.getint("testing", "n_processes")

dataset_path = pathlib.Path("./data")
new_dataset_path = dataset_path.joinpath("datasets_for_comparison")


def get_artificial_ds():
    data_sets = [f.stem.split("_")[0] for f in new_dataset_path.glob("*.csv")]

    all_artifical_dataset_fns = list(Counter([fn for fn in data_sets if re.match(r"\d+", fn)]).keys())
    return all_artifical_dataset_fns


def get_adult_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("adult*.csv")]).keys())
    return data_sets

def get_bank_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("bank*.csv")]).keys())
    return data_sets
def get_vote_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("vote*.csv")]).keys())
    return data_sets

def run_test_sklearn_models_on_dataset(dataset_name: str, model_name: str, model_cls, parameter_grid: Dict,
                                       categotrical_to_onehot=False) -> Dict:

    if categotrical_to_onehot:
        one_hot = "_one_hot"
    else:
        one_hot = ""

    train_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_train{one_hot}.csv"), index_col=0).to_numpy()
    val_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_val{one_hot}.csv"), index_col=0).to_numpy()
    test_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_test{one_hot}.csv"), index_col=0).to_numpy()

    best_val_score = float("-inf")
    best_params = None
    best_model = None
    for values in tqdm(product(*parameter_grid.values()), total=len(list(product(*parameter_grid.values()))),
                       file=sys.stdout):
        params = dict(zip(parameter_grid.keys(), values))
        try:
            model = model_cls(**params, random_state=SEED)
        except TypeError:
            model = model_cls(**params)

        model.fit(train_data[:, :-1], train_data[:, -1])
        val_score = model.score(val_data[:, :-1], val_data[:, -1])

        if val_score > best_val_score:
            best_val_score = val_score
            best_params = params
            best_model = model

    # print(best_params)
    # print(best_val_score)

    test_score = best_model.score(test_data[:, :-1], test_data[:, -1])
    # print(test_score)

    return {"model_cls": model_name, "dataset": dataset_name, "is_one_hot": categotrical_to_onehot,
            "best_params": best_params, "test_score": test_score, "val_score": best_val_score}


def run_test_for_sklearn_models(datasets_names: List[str], model_name: str, model_cls, parameter_grid: Dict,
                                categotrical_to_onehot=False) -> List:
    t = partial(run_test_sklearn_models_on_dataset, model_name=model_name,
                model_cls=model_cls,
                parameter_grid=parameter_grid,
                categotrical_to_onehot=categotrical_to_onehot
                )
    with Pool(N_PROCESSES) as p:
        results = p.map(t, dataset_names)

    return results


if __name__ == "__main__":

    decision_tree_param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha": [0.0, 0.0001, 0.001, 0.01]
    }

    random_forest_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    log_reg_param_grid = {
        "solver": ["liblinear", "lbfgs", "saga"],
        "C": [0.01, 0.1, 1, 10, 100],
        "max_iter": [100, 200, 500],
        "class_weight": [None, "balanced"],
    }

    nn_param_grid = {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "p": [1, 2]
    }

    # dataset_names = get_artificial_ds()
    # run_info = [
    #     (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
    #      "results_artificial_datasets_decision_tree.json"),
    #     (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
    #      "results_artificial_datasets_random_forest.json"),
    #     (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
    #      "results_artificial_datasets_logistic_regression.json"),
    #     (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
    #      "results_artificial_datasets_nearest_neighbor.json"),
    # ]
    #
    # for info in run_info:
    #     results = run_test_for_sklearn_models(datasets_names=info[0], model_name=info[1], model_cls=info[2], parameter_grid=info[3],
    #                        categotrical_to_onehot=False)
    #     with open(info[4], "w") as f:
    #         json.dump(results, f)

    dataset_names = []#get_adult_ds()
    #dataset_names.extend(get_bank_ds())
    dataset_names.extend(get_vote_ds())
    run_info = [
        (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
         "results_big_datasets_decision_tree.json", False),
        # (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
        #  "results_big_datasets_random_forest.json", False),
        # (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
        #  "results_big_datasets_logistic_regression.json", False),
        # (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
        #  "results_big_datasets_nearest_neighbor.json", False),
    ]

    for info in run_info:
        results = run_test_for_sklearn_models(datasets_names=info[0], model_name=info[1], model_cls=info[2], parameter_grid=info[3],
                                              categotrical_to_onehot=info[5])
        with open(info[4], "w") as f:
            json.dump(results, f)

    dataset_names = []#get_adult_ds()
    #dataset_names.extend(get_bank_ds())
    dataset_names.extend(get_vote_ds())
    run_info = [
        (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
         "results_big_datasets_decision_tree_one_hot.json", True),
        # (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
        #  "results_big_datasets_random_forest_one_hot.json", True),
        # (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
        #  "results_big_datasets_logistic_regression_one_hot.json", True),
        # (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
        #  "results_big_datasets_nearest_neighbor_one_hot.json", True),
    ]

    for info in run_info:
        results = run_test_for_sklearn_models(datasets_names=info[0], model_name=info[1], model_cls=info[2], parameter_grid=info[3],
                                              categotrical_to_onehot=info[5])
        with open(info[4], "w") as f:
            json.dump(results, f)
