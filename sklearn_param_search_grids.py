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