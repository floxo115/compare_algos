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

config_parser = configparser.ConfigParser()
config_parser.read("config.ini", encoding="utf-8")
SEED = config_parser.getint("testing", "seed")
N_PROCESSES = config_parser.getint("testing", "n_processes")

dataset_path = pathlib.Path("./data")
new_dataset_path = dataset_path.joinpath("datasets_for_comparison")


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
    prods = [juice.multiply(*inputs) for _ in range(2)]
    ns = juice.summate(*prods, num_node_blocks=1)

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
        # print(batch.shape)
        # each known variable is set to False
        missing_mask = torch.zeros_like(batch, dtype=torch.bool).to(device)
        missing_mask[:, -1] = 1
        # print(missing_mask)
        # print(missing_mask.shape)
        # print(batch.shape)
        # print(batch)
        # compute P(class| x_1...x_10)
        outputs = conditional(
            model,
            data=batch,
            missing_mask=missing_mask,
            target_vars=[n_variables - 1],
        )

        # get class with maximum probability
        preds = outputs.argmax(dim=2).flatten()
        # truth = batch[:, -1]

        # compute correct count accuracy
        acc += (preds == truth).sum().item()

    # compute accuracy
    return acc / n_samples


# df = pd.read_csv("./data/datasets_for_comparison/5_train.csv", index_col=0)
#
# print(df)
#
# device = (
#     torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# )
#
# dnp = df.to_numpy()
# X= dnp
# X = torch.tensor(X, dtype=torch.long, device=device)
#
# pc = create_probabilistic_circuit(X.shape[1], [2]*X.shape[1], 10)
#
# print(pc(X)[:5])
# train_loader = DataLoader(X, shuffle=True, batch_size=512)
# print(get_test_score(pc, train_loader))
#
# optimizer = CircuitOptimizer(pc, lr=0.1, pseudocount=0.1, method="EM")
#
# for epoch in range(1, 1000 + 1):
#     # t0 = time.time()
#     #print(epoch)
#     train_ll = 0.0
#     for batch in train_loader:
#         x = batch.to(device)
#
#         # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
#         optimizer.zero_grad()
#
#         # Forward pass
#         lls = pc(x)
#
#         # Backward pass
#         lls.mean().backward()
#
#         train_ll += lls.mean().detach().cpu().numpy().item()
#
#         # Perform a mini-batch EM step
#         optimizer.step()
#
#     # train_ll /= len(train_loader)
#
#     # print(
#     #     f"[Epoch {epoch}/{100}][train LL: {train_ll:.2f}]",
#     #     end="\r",
#     # )
# print(pc(X)[:5])
# print(get_test_score(pc, train_loader))

def run_pc_test(class_descr, lr, ds_name, depth):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    torch.manual_seed(SEED)

    train_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_train.csv"), index_col=0).to_numpy(),
                              dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    val_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_val.csv"), index_col=0).to_numpy())
    val_loader = DataLoader(TensorDataset(val_data), batch_size=32, shuffle=False)
    test_data = torch.tensor(pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_test.csv"), index_col=0).to_numpy())
    test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)

    num_cat = pd.read_csv(new_dataset_path.joinpath(f"{ds_name}_num_cats.csv"), header=None).to_numpy().flatten()
    print(f"num_cat: {num_cat}")

    pc = create_probabilistic_circuit(len(num_cat), num_cat, depth)
    optimizer = CircuitOptimizer(pc, lr=lr, method="EM")

    best_val_score = float("-inf")
    best_model_state_dict = None

    for epoch in tqdm(range(1, 500 + 1)):
        # t0 = time.time()
        # print(epoch)
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
            optimizer.zero_grad()

            # Forward pass
            lls = pc(x)

            # Backward pass
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            # Perform a mini-batch EM step
            optimizer.step()

        score = get_test_score(pc, val_loader)
        if score >= best_val_score:
            best_val_score = score
            best_model_state_dict = pc.state_dict()
            print(f"saving best model at epoch {epoch}, new best score {best_val_score}")

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


datasets = []
#datasets.extend(get_artificial_ds())
#datasets.extend(get_adult_ds())
#datasets.extend(get_bank_ds())
datasets.extend(get_vote_ds())
print(datasets)
results = []
for ds in datasets:
    res = run_pc_test("pc", 0.0001, ds, depth=10)
    print(res)
    results.extend(res)

for ds in datasets:
    res = run_pc_test("pc", 0.01, ds, depth=2)
    print(res)
    results.extend(res)

json.dump(results, open("results_probabilistic_circuits_tests.json", "w"))
