import configparser
import pathlib
import re
import sys
from collections import defaultdict
from functools import partial
from itertools import product
from pprint import pprint
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from multiprocessing import Pool

config_parser = configparser.ConfigParser()
config_parser.read("config.ini", encoding="utf-8")
SEED = config_parser.getint("testing", "seed")
N_PROCESSES = config_parser.getint("testing", "n_processes")

dataset_path = pathlib.Path("./data")
new_dataset_path = dataset_path.joinpath("datasets_for_comparison")

def get_artificial_ds():
    data_sets = [f.stem.split("_")[0] for f in new_dataset_path.glob("*.csv")]

    all_artifical_dataset_fns = [fn for fn in data_sets if re.match(r"\d+", fn)]
    return all_artifical_dataset_fns

def run_test(dataset_name: str, model_name: str,  model_cls, parameter_grid: Dict, categotrical_to_onehot=False) -> Dict:
    train_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_train.csv"), index_col=0).to_numpy()
    val_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_val.csv"), index_col=0).to_numpy()
    test_data = pd.read_csv(new_dataset_path.joinpath(f"{dataset_name}_test.csv"), index_col=0).to_numpy()

    best_val_score = float("-inf")
    best_params = None
    best_model = None
    for values in tqdm(product(*parameter_grid.values()), total=len(list(product(*parameter_grid.values()))), file=sys.stdout):
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

    return {"model_cls": model_name, "dataset": dataset_name, "is_one_hot": categotrical_to_onehot, "best_params": best_params, "test_score": test_score, "val_score": best_val_score}


if __name__ == "__main__":
    dataset_names = get_artificial_ds()

    decision_tree_param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha": [0.0, 0.0001, 0.001, 0.01]
    }


    t = partial(run_test, model_name="DecisionTreeClassifier",
                model_cls=DecisionTreeClassifier,
                parameter_grid=decision_tree_param_grid)
    with Pool(N_PROCESSES) as p:
        results = p.map(t, dataset_names)

    with open("results_artificial_datasets_decision_tree.json", "w") as f:
        json.dump(results, f)

    random_forest_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    t = partial(run_test, model_name="RandomForest",
                model_cls=RandomForestClassifier,
                parameter_grid=random_forest_param_grid)
    with Pool(N_PROCESSES) as p:
        results = p.map(t, dataset_names)
    with open("results_artificial_datasets_random_forest.json", "w") as f:
        json.dump(results, f)

    log_reg_param_grid = {
        "solver": ["liblinear", "lbfgs", "saga"],
        "C": [0.01, 0.1, 1, 10, 100],
        "max_iter": [100, 200, 500],
        "class_weight": [None, "balanced"],
    }

    t = partial(run_test, model_name="LogisticRegression",
                model_cls=LogisticRegression,
                parameter_grid=log_reg_param_grid)
    with Pool(N_PROCESSES) as p:
        results = p.map(t, dataset_names)
    with open("results_artificial_datasets_logistic_regression.json", "w") as f:
        json.dump(results, f)


    nn_param_grid = {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "p": [1, 2]
    }

    t = partial(run_test, model_name="NearestNeighbor",
                model_cls=KNeighborsClassifier,
                parameter_grid=nn_param_grid)
    with Pool(N_PROCESSES) as p:
        results = p.map(t, dataset_names)
    with open("results_artificial_datasets_nearest_neighbor.json", "w") as f:
        json.dump(results, f)