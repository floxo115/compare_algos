import json
import sys
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

from config import config
from sklearn_param_search_grids import decision_tree_param_grid, random_forest_param_grid, log_reg_param_grid, \
    nn_param_grid
from utils import get_artificial_ds, get_adult_ds, get_bank_ds, get_vote_ds, get_secondary_mushrooms_ds

SEED = config["seed"]
N_PROCESSES = config["n_processes"]
dataset_path = config["dataset_path"]
new_dataset_path = config["new_dataset_path"]


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

    test_score = best_model.score(test_data[:, :-1], test_data[:, -1])

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
    dataset_names = get_artificial_ds()
    run_info = [
        # (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
        #  "results_artificial_datasets_decision_tree.json"),
        # (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
        #  "results_artificial_datasets_random_forest.json"),
        # (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
        #  "results_artificial_datasets_logistic_regression.json"),
        # (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
        #  "results_artificial_datasets_nearest_neighbor.json"),
    ]

    for info in run_info:
        results = run_test_for_sklearn_models(datasets_names=info[0], model_name=info[1], model_cls=info[2],
                                              parameter_grid=info[3],
                                              categotrical_to_onehot=False)
        with open(info[4], "w") as f:
            json.dump(results, f)

    dataset_names = []
    dataset_names.extend(get_adult_ds())
    dataset_names.extend(get_bank_ds())
    dataset_names.extend(get_vote_ds())
    # dataset_names.extend(get_secondary_mushrooms_ds())
    run_info = [
        # (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
        #  "results_secondary_mushrooms_decision_tree.json", False),
        # (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
        #  "results_secondary_mushrooms_random_forest.json", False),
        # (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
        #  "results_secondary_mushrooms_logistic_regression.json", False),
        # (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
        #  "results_secondary_mushrooms_nearest_neighbor.json", False),

        (dataset_names, "DecisionTreeClassifier", DecisionTreeClassifier, decision_tree_param_grid,
         "results_datasets_decision_tree_one_hot.json", True),
        (dataset_names, "RandomForestClassifier", RandomForestClassifier, random_forest_param_grid,
         "results_datasets_random_forest_one_hot.json", True),
        (dataset_names, "LogisticRegression", LogisticRegression, log_reg_param_grid,
         "results_datasets_logistic_regression_one_hot.json", True),
        (dataset_names, "KNeighborsClassifier", KNeighborsClassifier, nn_param_grid,
         "results_datasets_nearest_neighbor_one_hot.json", True),
    ]

    for info in run_info:
        results = run_test_for_sklearn_models(datasets_names=info[0], model_name=info[1], model_cls=info[2],
                                              parameter_grid=info[3],
                                              categotrical_to_onehot=info[5])
        with open(info[4], "w") as f:
            json.dump(results, f)

# run everything for secondary-mushrooms check
# run one-hot for everything else after that and rename the outputs check
# run mushroom for pc done
# make flat pc and run tests
