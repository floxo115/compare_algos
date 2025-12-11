import configparser
import json
import pathlib
import re
from collections import Counter

import pandas as pd
import pyjuice as juice
import torch
from pyjuice.optim import CircuitOptimizer
from pyjuice.queries import conditional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import config
from utils import get_artificial_ds, get_adult_ds, get_bank_ds, get_vote_ds, get_secondary_mushrooms_ds

SEED = config["seed"]
N_PROCESSES = config["n_processes"]
dataset_path = config["dataset_path"]
new_dataset_path = config["new_dataset_path"]


def create_probabilistic_circuit(inp_len, cat_nums, depth):
    assert inp_len == len(cat_nums)
    assert depth >= 1

    # define the model
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    inputs = [
        juice.inputs(
            i, num_node_blocks=8, dist=juice.distributions.Categorical(num_cats=cat_nums[i])
        )
        for i in range(inp_len)
    ]

    prods = [juice.multiply(*inputs) for _ in range(5)]
    ns = juice.summate(*prods, num_node_blocks=5)

    for depth in range(1, depth):
        products = juice.multiply(*inputs)
        ns = juice.summate(juice.multiply(ns), products, num_node_blocks=1)

    ns.init_parameters()

    pc = juice.compile(ns)
    pc.to(device)

    return pc


def get_test_score(model, val_loader):
    """computes accuracy of the given model on the test set"""
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    acc = 0.0
    n_samples = 0
    for batch in val_loader:
        # get samples of current batch and add them to the overall count
        n_samples += batch[0].shape[0]
        n_variables = batch[0].shape[1]
        batch = batch[0].to(device)
        truth = batch[:, -1].clone().to(device)
        batch[:, -1] = -100

        missing_mask = torch.zeros_like(batch, dtype=torch.bool).to(device)
        missing_mask[:, -1] = 1
        outputs = conditional(
            model,
            data=batch.to(torch.long),
            missing_mask=missing_mask,
            target_vars=[n_variables - 1],
        )

        preds = outputs.argmax(dim=2).flatten()

        acc += (preds == truth).sum().item()

    return acc / n_samples


def run_pc_test(class_descr, lr, ds_name, depth):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    torch.manual_seed(SEED)

    train_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_train.csv"), index_col=0).to_numpy(),
                              dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)
    val_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_val.csv"), index_col=0).to_numpy())
    val_loader = DataLoader(TensorDataset(val_data), batch_size=100000, shuffle=False)
    test_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_test.csv"), index_col=0).to_numpy())
    test_loader = DataLoader(TensorDataset(test_data), batch_size=10000, shuffle=False)

    num_cat = pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_num_cats.csv"), header=None).to_numpy().flatten()
    print(f"num_cat: {num_cat}")

    pc = create_probabilistic_circuit(len(num_cat), num_cat, depth)

    optimizer = CircuitOptimizer(pc, lr=lr, method="EM")

    best_val_score = float("-inf")
    best_model_state_dict = None

    no_increase_since = 0
    for epoch in tqdm(range(1, 5000 + 1)):
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()
            optimizer.step()

        score = get_test_score(pc, val_loader)
        if score >= best_val_score + 10e-16:
            no_increase_since = 0
            best_val_score = score
            best_model_state_dict = pc.state_dict()
            print(f"saving best model at epoch {epoch}, new best score {best_val_score}")
        else:
            no_increase_since += 1
            if no_increase_since >= 100:
                break

    pc.load_state_dict(best_model_state_dict)
    test_score = get_test_score(pc, test_loader)

    return [
        {
            "model_cls": class_descr,
            "dataset": ds_name,
            "is_one_hot": False,
            "lr": lr,
            "test_score": test_score,
            "val_score": best_val_score
        }
    ]


datasets = []
# datasets.extend(get_artificial_ds())
# datasets.extend(get_adult_ds())
# datasets.extend(get_bank_ds())
# datasets.extend(get_vote_ds())
datasets.extend(get_secondary_mushrooms_ds())
print(datasets)
results = []
for ds in datasets:
    res = run_pc_test("HCLT w. hidden vars: 13", 0.01, ds, depth=1)
    print(res)
    results.extend(res)

for ds in datasets:
    res = run_pc_test("HCLT w. hidden vars: 141", 0.01, ds, depth=30)
    print(res)
    results.extend(res)

for ds in datasets:
    res = run_pc_test("HCLT w. hidden vars: 141", 0.1, ds, depth=10)
    print(res)
    results.extend(res)

json.dump(results, open("res/results_hclt_13_hidden_vars_all_datasets_tests.json", "w"))
